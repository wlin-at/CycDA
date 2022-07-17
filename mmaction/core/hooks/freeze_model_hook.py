from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class FreezeModelHook(Hook):

    def __init__(self, module_name_list = None):
        self.module_name_list = module_name_list

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        if len(self.module_name_list) > 0:
            print(f'Freezing {self.module_name_list} ...')
            for module_name in self.module_name_list:
                # runner.model.module.backbone._modules[module_name].modules
                for param in runner.model.module.backbone._modules[module_name].parameters():
                    param.requires_grad = False

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
