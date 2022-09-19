import kornia as K


def construct_list_of_kornia_augs(
        params,
):
    """
    Args:
        params: list with kornia augs and params example:
        [
                {kornia_aug: ColorJitter, p: 0.5, brightness: 0.5 , contrast: 0.5, saturation: 0.5, hue: 0.5},
                {kornia_aug: RandomEqualize, p: 0.5},
                {kornia_aug: RandomGrayscale, p: 0.5},
                {kornia_aug: RandomPosterize, p: 0.5},
                {kornia_aug: RandomSolarize, p: 0.5},
        ]

    Returns:
        list of kornia augs
    """
    augs = []
    for value in params:
        print(value)
        kornia_aug = value.pop('kornia_aug')
        kornia_aug = getattr(K.augmentation, kornia_aug, None)
        kornia_params = {}
        for key, param in value.items():
            if isinstance(param, list):
                param = tuple(param)
            kornia_params[key] = param
        augs.append(kornia_aug(**kornia_params))
    return augs


def construct_kornia_aug(
        params,
        same_on_batch=True,
        keepdim=True,
        random_apply=1
):
    """
        Args:
            params: see construct_list_of_kornia_augs
            same_on_batch: same aug along batch
            keepdim: keepdim
            random_apply: randomly select a sublist (order agnostic) of args to apply transformation.
                        If int, a fixed number of transformations will be selected.
                        If (a,), x number of transformations (a <= x <= len(args)) will be selected.
                        If (a, b), x number of transformations (a <= x <= b) will be selected.
                        If True, the whole list of args will be processed as a sequence in a random order.
                        If False, the whole list of args will be processed as a sequence in original order.

        Returns:
            list of kornia augs
        """

    augs_list = construct_list_of_kornia_augs(params)
    return K.augmentation.ImageSequential(*augs_list,
                                          same_on_batch=same_on_batch,
                                          keepdim=keepdim,
                                          random_apply=random_apply)
