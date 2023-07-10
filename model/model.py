from model.resnet import Resnet50_fc, Resnet50_Siam
from model.transformer import vit_Siam
from byol_pytorch import BYOL
from torchvision import models
import timm.models
import torch

def build_net(args):
    if args.backbone == 'resnet':
        if args.SSL == 'None':
            #model = models.resnet18(pretrained=True)
            model = models.resnet50(pretrained=True)
            model.fc = torch.nn.Linear(2048, 2)
            #model.fc = torch.nn.Identity()

        elif args.SSL == 'SimSiam':
            # model = Resnet50_fc(2)
            model = Resnet50_Siam(2)
       
    elif args.backbone == 'vit_B_32':
        fea_dim = 768

        if args.SSL == 'None':
            # Transformer ##
            model = timm.create_model(
                #'vit_base_patch32_224_in21k',
                'vit_base_patch32_224',
                pretrained=True,
                num_classes=2
                )
        elif args.SSL == 'SimSiam':
            model = vit_Siam(2, fea_dim)
        elif args.SSL == 'BYOL':
            vit = timm.create_model(
                #'vit_base_patch32_224_in21k',
                'vit_base_patch32_224',
                pretrained=True,
                num_classes=2
            )
            model = BYOL(
                vit,
                image_size=256,
                pre_class_dim=fea_dim,
                hidden_layer='pre_logits', ### 해당 layer 찾아서 그 결과를 representation으로 활용
                use_momentum=True  # turn off momentum in the target encoder
            )

    return model
