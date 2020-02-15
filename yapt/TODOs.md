# TODOs
- [] make sure that type dow not change in args and also handle missing params
- [] discuss how to handle nograd in validation
- [] use collect dataset for both training and validation
- [] call on_epoch_end to decide what to do with outputs (collate)
- [] call on_validation_end to decide what to do with outputs (collate)
- [] rename stats in tb_scalars, tb_images and write doc about how to handle it.
- [] add tqdm_running dict and tqdm_final
- [] create_episodes for meta datasets. where should it be called in yapt?
- [] check schedulers.step best practices
- [] create .secret and handle sacred mongodb (username is default from userid) https://gitlab.com/airlab-404/server-guidelines/-/wikis/home

- [x] add flag to disable tqdm
- [x] modify vae models for schedulers and optimizers
