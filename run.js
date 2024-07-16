async function run(textInput, itemsQuery) {
  let tokenizer;
  let session;
  let transformers;
  let featureSetId;
  const default_query = {
    filter: { $and: [{ hidden: false }, { type: "file" }] },
  };
  const dataset = await dl.datasets.get();

  try {
    textInput = textInput["Text Box"];
  } catch (e) {
    dl.sendEvent({
      name: "app:toastMessage",
      payload: {
        message: "For Meta CLIP FeatureSet input text is required",
        type: "error",
      },
    });
    return default_query;
  }

  try {
    const item = await dl.items.getByName("/meta_clip_feature_set.json", {binaries: true});
    featureSetId = item.metadata.system.meta_clip_feature_set_id;
  } catch (e) {
    console.log(e);
    dl.sendEvent({
      name: "app:toastMessage",
      payload: {
        message:
          "Meta CLIP FeatureSet does not exist for this project, running pre-process",
        type: "warning",
      },
    });
    const execution = await dl.executions.create({
      functionName: "extract_dataset",
      serviceName: "meta-clip-extraction",
      input: { dataset: { dataset_id: dataset.id }, query: null },
    });
    return default_query;
  }

  const query_feature = {
    filter: {
      $and: [{ hidden: false }, { type: "file" }],
    },
    resource: "items",
    join: {
      on: {
        resource: "feature_vectors",
        local: "entityId",
        forigen: "id",
      },
      filter: {
        featureSetId: featureSetId,
      },
    },
  };

  const items_count = dataset.itemsCount;
  const items_with_feature_count = await dl.items.countByQuery(query_feature);
  if (items_count !== items_with_feature_count) {
    dl.sendEvent({
      name: "app:toastMessage",
      payload: {
        message:
          "Feature extraction was not run on entire dataset, please run again!",
        type: "warning",
      },
    });
  }
  console.log("loading dependencies");

  async function fetchONNXFile() {
    // "https://media.githubusercontent.com/media/dataloop-ai-apps/meta-test/main/weights/metaclip_text_encoder.onnx";
    let baseName =
      "https://media.githubusercontent.com/media/dataloop-ai-apps/meta-test/main/weights/metaclip_text_encoder";
    let extension = ".onnx";
    let urls = [];
    for (let i = 0; i <= 24; i++) {
      const postfix = i.toString().padStart(2, "0");
      urls.push(`${baseName}_${postfix}${extension}`);
    }

    const chunkPromises = urls.map((url) =>
      fetch(url).then((response) => response.arrayBuffer())
    );

    const chunks = await Promise.all(chunkPromises);

    const totalSize = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
    const mergedBuffer = new ArrayBuffer(totalSize);
    const uint8View = new Uint8Array(mergedBuffer);

    // Step 3: Copy chunks into the merged buffer
    let offset = 0;
    for (const chunk of chunks) {
      uint8View.set(new Uint8Array(chunk), offset);
      offset += chunk.byteLength;
    }
    return mergedBuffer;
  }

  async function loadDependencies() {
    try {
      transformers = await import(
        "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.1"
      );
      transformers.env.allowLocalModels = false;
      transformers.env.remoteHost = "https://huggingface.co/";
      transformers.env.remotePathTemplate =
        "Xenova/clip-vit-base-patch32/resolve/main";
      // ort = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.8.0");
      ort = await import(
        "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/ort.min.js"
      );
      ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

      tokenizer = await transformers.AutoTokenizer.from_pretrained(
        "Xenova/clip-vit-base-patch32"
      );

      session = await ort.InferenceSession.create(await fetchONNXFile());
    } catch (e) {
      console.log(e);
      return default_query;
    }
  }
  await loadDependencies();

  let texts = [textInput];
  let textInputs = tokenizer(texts, {
    padding: true,
    max_length: 77,
    truncation: true,
  });
  let bigIntArray = textInputs.input_ids[0].data; // Assuming this is an array of BigInt
  let numberArray = Array.from(bigIntArray, (bigInt) => Number(bigInt)); // Convert each BigInt to Number
  let int32Array = new Int32Array(numberArray); // Create Int32Array from number array

  let inputTensor = new ort.Tensor(int32Array, [1, 77]);
  const feeds = { image: inputTensor };
  let results = await session.run(feeds);
  let vector = results.text_features.data;
  console.log(vector);

  let query = {
    filter: { $and: [{ hidden: false }, { type: "file" }] },
    page: 0,
    pageSize: 1000,
    resource: "items",
    join: {
      on: {
        resource: "feature_vectors",
        local: "entityId",
        forigen: "id",
      },
      filter: {
        value: {
          $euclid: {
            input: Array.from(vector),
            $euclidSort: { eu_dist: "ascending" },
          },
        },
        featureSetId: featureSetId,
      },
    },
  };
  console.log(query);
  return query;
}
