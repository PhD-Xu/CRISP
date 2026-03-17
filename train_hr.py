import os
import random
import torch
random.seed(42)
#os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [1]))
torch.manual_seed(42)
import typer

import pytorch_lightning as pl
import torchvision.transforms as transforms
from augment.auto_augment import AutoAugment
from transformers import BertTokenizerFast, ViltImageProcessor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from model.RSVQA_model import VQAModel
from dataloader.VQALoader_HR import VQALoader

from pytorch_lightning.tuner import Tuner


def main(num_workers: int = 16,
         ratio_images_to_use: int = 1,
         sequence_length: int = 40,
         num_epochs: int = 40,
         batch_size: int = 50,
         lr: float = 1e-3,
         grad_clip_val: float = 1.0,
         Dataset='HR'):

    data_path = 'YOUE/PATH'

    HR_questionsJSON = os.path.join(data_path, 'USGS_split_train_questions.json')
    HR_answersJSON = os.path.join(data_path, 'USGS_split_train_answers.json')
    HR_imagesJSON = os.path.join(data_path, 'USGS_split_train_images.json')
    HR_questionsvalJSON = os.path.join(data_path, 'USGS_split_val_questions.json')
    HR_answersvalJSON = os.path.join(data_path, 'USGS_split_val_answers.json')
    HR_imagesvalJSON = os.path.join(data_path, 'USGS_split_val_images.json')
    # HR_questionstestJSON = os.path.join(data_path, 'USGS_split_test_questions.json')
    # HR_answerstestJSON = os.path.join(data_path, 'USGS_split_test_answers.json')
    # HR_imagestestJSON = os.path.join(data_path, 'USGS_split_test_images.json')
    HR_images_path = os.path.join(data_path, 'Data/')


    #-------------------------------- test1--------------------------------------
    HR_questionstestJSON_1 = os.path.join(data_path, 'USGS_split_test_questions.json')
    HR_answerstestJSON_1 = os.path.join(data_path, 'USGS_split_test_answers.json')
    HR_imagestestJSON_1 = os.path.join(data_path, 'USGS_split_test_images.json')
    #--------------------------------test1--------------------------------------
    
    #--------------------------------test2--------------------------------------
    HR_questionstestJSON_2 = os.path.join(data_path, 'USGS_split_test_phili_questions.json')
    HR_answerstestJSON_2 = os.path.join(data_path, 'USGS_split_test_phili_answers.json')
    HR_imagestestJSON_2 = os.path.join(data_path, 'USGS_split_test_phili_images.json')
    #--------------------------------test2--------------------------------------

    tokenizer = BertTokenizerFast.from_pretrained('dandelin/vilt-b32-mlm')
    image_processor = ViltImageProcessor(do_resize=True, image_std=[0.229, 0.224, 0.225], image_mean=[0.485, 0.456, 0.406], do_rescale=True, do_normalize=True, size=512, size_divisor=32)
    
    if Dataset == 'HR':
        model = VQAModel(batch_size=batch_size, lr=lr, number_outputs=98)
    else:
        model = VQAModel(batch_size=batch_size, lr=lr, number_outputs=9)

    transform_train = [
            transforms.RandomHorizontalFlip(),
        ]
    transform_train.append(AutoAugment())
    transform_train = transforms.Compose(transform_train)
    # loader for the training data
    HR_data_train = VQALoader(HR_images_path,
                              HR_imagesJSON,
                              HR_questionsJSON,
                              HR_answersJSON,
                              tokenizer=tokenizer,
                              image_processor=image_processor,
                              Dataset='HR',
                              train=True,
                              sequence_length=sequence_length,
                              ratio_images_to_use=ratio_images_to_use,
                              transform=transform_train)
    
    HR_train_loader = torch.utils.data.DataLoader(HR_data_train, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers)
    
    # loader for the validation data
    HR_data_val = VQALoader(HR_images_path,
                            HR_imagesvalJSON,
                            HR_questionsvalJSON,
                            HR_answersvalJSON,
                            tokenizer=tokenizer,
                            image_processor=image_processor,
                            Dataset='HR',
                            train=False,
                            ratio_images_to_use=ratio_images_to_use,
                            sequence_length=sequence_length,)
    
    HR_val_loader = torch.utils.data.DataLoader(HR_data_val, batch_size=batch_size, shuffle=False,
                                                num_workers=num_workers)
    
    #---------------------------------------------------------------------------------------------------
    HR_data_test_1 = VQALoader(HR_images_path,
                             HR_imagestestJSON_1,
                             HR_questionstestJSON_1,
                             HR_answerstestJSON_1,
                             tokenizer=tokenizer,
                            image_processor=image_processor,
                            Dataset='HR',
                            train=False,
                            ratio_images_to_use=ratio_images_to_use,
                            sequence_length=sequence_length,)

    HR_test_loader_1 = torch.utils.data.DataLoader(HR_data_test_1, batch_size=batch_size, shuffle=False,
                                                num_workers=num_workers)
    #---------------------------------------------------------------------------------------------------

    #---------------------------------------------------------------------------------------------------
    HR_data_test_2 = VQALoader(HR_images_path,
                             HR_imagestestJSON_2,
                             HR_questionstestJSON_2,
                             HR_answerstestJSON_2,
                             tokenizer=tokenizer,
                            image_processor=image_processor,
                            Dataset='HR',
                            train=False,
                            ratio_images_to_use=ratio_images_to_use,
                            sequence_length=sequence_length,)

    HR_test_loader_2 = torch.utils.data.DataLoader(HR_data_test_2, batch_size=batch_size, shuffle=False,
                                                num_workers=num_workers)
    #---------------------------------------------------------------------------------------------------

    

    wandb_logger = WandbLogger(project='YOUE/PATH')

    # specify how to checkpoint
    checkpoint_callback = ModelCheckpoint(save_top_k=2,
                                          monitor="valid_acc",
                                          save_weights_only=True,
                                          mode="max",
                                          dirpath='YOUE/PATH',
                                          filename=f"{{epoch}}_{{valid_acc:.5f}}")

    # early stopping
    early_stopping = EarlyStopping(monitor="valid_acc", patience=15, mode="max")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(devices=1,
                         accelerator='cuda',
                         fast_dev_run=False,
                         precision='16-mixed',
                         max_epochs=num_epochs,
                         logger=wandb_logger,
                         #strategy='ddp_find_unused_parameters_true',
                         num_sanity_val_steps=0,
                         gradient_clip_val=grad_clip_val,
                         gradient_clip_algorithm="norm",
                         callbacks=[checkpoint_callback, early_stopping, lr_monitor])

    trainer.fit(model, train_dataloaders=HR_train_loader, val_dataloaders=HR_val_loader)

    trainer.test(model, dataloaders=HR_test_loader_1)

    trainer.test(model, dataloaders=HR_test_loader_2)


if __name__ == "__main__":
    typer.run(main)
