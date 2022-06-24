import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import verify_results

from adaptor.engine import AdaptORTrainer, D2Trainer
from adaptor.config import get_cfg
from adaptor.modeling import convert_splitgn_model


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        trainer = AdaptORTrainer if cfg.UDA.ENABLE else D2Trainer
        model = trainer.build_model(cfg)
        if cfg.MODEL.USE_SPLIT_GROUP_NORM:
            model = convert_splitgn_model(
                model, inference_type=cfg.MODEL.SPLIT_GN_INFER_TYPE
            )

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = AdaptORTrainer(cfg) if cfg.UDA.ENABLE else D2Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
