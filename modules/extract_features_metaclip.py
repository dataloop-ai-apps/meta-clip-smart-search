from PIL import Image, ImageFile
import dtlpy as dl
import logging
import torch
import time
from transformers import CLIPProcessor, CLIPModel
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json

logger = logging.getLogger('[META-CLIP-SEARCH]')
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MetaClipExtractor(dl.BaseServiceRunner):
    def __init__(self, project=None):
        if project is None:
            project = self.service_entity.project
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CLIPModel.from_pretrained("facebook/metaclip-b32-400m")
        self.processor = CLIPProcessor.from_pretrained("facebook/metaclip-b32-400m")
        self.model.to('cpu')

        self.feature_set = None
        self.feature_vector_entities = list()
        self.create_feature_set(project=project)

    def create_feature_set(self, project: dl.Project):
        try:
            feature_set = project.feature_sets.get(feature_set_name='meta-clip-feature-set')
            logger.info(f'Feature Set found! name: {feature_set.name}, id: {feature_set.id}')
        except dl.exceptions.NotFound:
            logger.info('Feature Set not found. creating...')

            feature_set = project.feature_sets.create(name='meta-clip-feature-set',
                                                      entity_type=dl.FeatureEntityType.ITEM,
                                                      project_id=project.id,
                                                      set_type='meta-clip',
                                                      size=512)
            logger.info(f'Feature Set created! name: {feature_set.name}, id: {feature_set.id}')
            binaries = project.datasets._get_binaries_dataset()
            buffer = BytesIO()
            buffer.write(json.dumps({}, default=lambda x: None).encode())
            buffer.seek(0)
            buffer.name = "meta_clip_feature_set.json"
            feature_set_item = binaries.items.upload(
                local_path=buffer,
                item_metadata={
                    "system": {
                        "meta_clip_feature_set_id": feature_set.id
                    }
                }
            )
        self.feature_set = feature_set
        self.feature_vector_entities = [fv.entity_id for fv in self.feature_set.features.list().all()]

    def extract_item(self, item: dl.Item) -> dl.Item:
        if item.id in self.feature_vector_entities:
            logger.info(f'Item {item.id} already has feature vector')
            return item
        logger.info(f'Started on item id: {item.id}, filename: {item.filename}')
        tic = time.time()

        # Preprocess for Meta clip
        orig_image = Image.fromarray(item.download(save_locally=False, to_array=True))

        inputs = self.processor(images=orig_image, return_tensors='pt').to('cpu')

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        image_features = outputs[0].cpu().tolist()
        self.feature_set.features.create(value=image_features, entity=item)
        logger.info(f'Done. runtime: {(time.time() - tic):.2f}[s]')

        self.feature_vector_entities.append(item.id)
        return item

    def extract_dataset(self, dataset: dl.Dataset, query=None, progress=None):
        pages = dataset.items.list()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.extract_item, obj) for obj in pages.all()]
            done_count = 0
            previous_update = 0
            while futures:
                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                done_count += len(done)

                current_progress = done_count * 100 // pages.items_count

                if (current_progress // 10) % 10 > previous_update:
                    previous_update = (current_progress // 10) % 10
                    if progress is not None:
                        progress.update(progress=previous_update * 10)
                    else:
                        logger.info(f'Extracted {done_count} out of {pages.items_count} items')
        return dataset


if __name__ == "__main__":
    dl.setenv('rc')
    project = dl.projects.get(project_name='COCO ors')
    app = MetaClipExtractor(project=project)
    dataset = dl.datasets.get(dataset_id='5f4d13ba4a958a49a7747cd9')
    app.extract_dataset(dataset=dataset)
