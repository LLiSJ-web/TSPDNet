
def build(model_name, args):
    model = None
    if model_name == 'BaseUMamba-SOD': # 基线
        from BaseUMamba import get_BaseUMamba as BaseUMamba
        model = BaseUMamba(deep_supervision=True,
                           use_pretrain=True,
                           img_size=args.img_size,
                           dims=128,
                           pretrained_path=args.pretrained_path,
                           )
    elif model_name in ['TSPDNet-V-TSOD', 'TSPDNet-V-SOD']: # 最终版本
        from TSPDNet import bulid_model
        model = bulid_model(
            deep_supervision=True,
            use_pretrain=True,
            img_size=args.img_size,
            dims=128,
            depths=[2, 2, 2, 2],
            pretrained_path=args.pretrained_path
        )
    return model
    
