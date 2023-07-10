from model.resnet import Resnet50_fc, Resnet50_Siam, Resnet50_Barlow
from model.transformer import vit_Siam, vit_Barlow
from BYOL.byol_pytorch import BYOL
from moco import builder
from torchvision import models
import timm.models
import torch
from model.vits import vit_base

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
        elif args.SSL == 'Barlow Twins':
            model = Resnet50_Barlow(2)
        elif args.SSL == 'BYOL':
            resnet = models.resnet50(pretrained=True)
            ## BYOL
            model = BYOL(
                resnet,
                image_size=256,
                pre_class_dim=2048,
                hidden_layer='avgpool',
                use_momentum=True  # turn off momentum in the target encoder
            )
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
        elif args.SSL == 'Barlow Twins':
            model = vit_Barlow(2, fea_dim)
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
        elif args.SSL == "MoCov3":
            vit = timm.create_model(
                # 'vit_base_patch32_224_in21k',
                'vit_base_patch32_224',
                pretrained=True,
                num_classes=2
            )

            #vit.head = torch.nn.Identity()
            model = builder.MoCo_ViT(vit) #vit_base

    elif args.backbone == 'vit_B_16':
        fea_dim = 768

        if args.SSL == 'None':
            # Transformer ##
            model = timm.create_model(
                #'vit_base_patch32_224_in21k',
                'vit_base_patch16_224',
                pretrained=True,
                num_classes=2
                )
        elif args.SSL == 'SimSiam':
            model = vit_Siam(2, fea_dim)
        elif args.SSL == 'Barlow Twins':
            model = vit_Barlow(2, fea_dim)
        elif args.SSL == 'BYOL':
            vit = timm.create_model(
                #'vit_base_patch32_224_in21k',
                'vit_base_patch16_224',
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
        elif args.SSL == "MoCov3":
           pass

    elif args.backbone == 'vit_S_16':
        fea_dim = 384

        if args.SSL == 'None':
            # Transformer ##
            model = timm.create_model(
                #'vit_base_patch32_224_in21k',
                #'vit_base_patch16_224',
                'vit_small_patch16_224',
                pretrained=True,
                num_classes=2
                )
        elif args.SSL == 'SimSiam':
            model = vit_Siam(2, fea_dim)
        elif args.SSL == 'Barlow Twins':
            model = vit_Barlow(2, fea_dim)
        elif args.SSL == 'BYOL':
            vit = timm.create_model(
                #'vit_base_patch32_224_in21k',
                #'vit_base_patch16_224',
                'vit_small_patch16_224',
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
        elif args.SSL == "MoCov3":
           pass

    elif args.backbone == 'vit_T_16':
        fea_dim = 192

        if args.SSL == 'None':
            # Transformer ##
            model = timm.create_model(
                'vit_tiny_patch16_224',
                pretrained=True,
                num_classes=2
            )
        elif args.SSL == 'SimSiam':
            model = vit_Siam(2, fea_dim)
        elif args.SSL == 'Barlow Twins':
            model = vit_Barlow(2, fea_dim)
        elif args.SSL == 'BYOL':
            vit = timm.create_model(
                'vit_tiny_patch16_224',
                pretrained=True,
                num_classes=2
            )
            model = BYOL(
                vit,
                image_size=256,
                pre_class_dim=fea_dim,
                hidden_layer='pre_logits',  ### 해당 layer 찾아서 그 결과를 representation으로 활용
                use_momentum=True  # turn off momentum in the target encoder
            )

    elif args.backbone == 'vit_hybrid_T_16':
        fea_dim = 192

        if args.SSL == 'None':
            # Transformer ##
            model = timm.create_model(
                'vit_tiny_r_s16_p8_224',
                pretrained=True,
                num_classes=2
            )

        elif args.SSL == 'SimSiam':
            model = vit_Siam(2, fea_dim)
        elif args.SSL == 'Barlow Twins':
            model = vit_Barlow(2, fea_dim)
        elif args.SSL == 'BYOL':
            vit = timm.create_model(
                'vit_tiny_r_s16_p8_224',
                pretrained=True,
                num_classes=2
            )
            model = BYOL(
                vit,
                image_size=256,
                pre_class_dim=fea_dim,
                hidden_layer='pre_logits',  ### 해당 layer 찾아서 그 결과를 representation으로 활용
                use_momentum=True  # turn off momentum in the target encoder
            )

    return model