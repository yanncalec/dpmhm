# Obsolete functions
def _wav2feature_pipeline_obslt(ds, module_name, extractor:callable, *,
dc_kwargs:dict, ft_kwargs:dict,
splits:dict=None, sp_mode:str='uniform', sp_kwargs:dict={}):
	"""Transform a dataset of waveform to feature.
	"""
	# module = import_module('..'+module_name, __name__)
	module = import_module('dpmhm.datasets.'+module_name)
	early_mode = 'early' in sp_mode.split('+')
	uniform_mode = 'uniform' in sp_mode.split('+')

	compactor = module.DatasetCompactor(ds, **dc_kwargs)

	transformer = module.FeatureTransformer(compactor.dataset, extractor, **ft_kwargs)
	# df_split[k] = transformer.dataset_feature
	dw = transformer.dataset_windows

	if splits is None:
		return dw
	else:
		if early_mode:
			ds_split = utils.split_dataset(compactor.dataset, splits,
			labels=None if uniform_mode else compactor.label_dict.keys(),
			**sp_kwargs
			)
			# df_split = {}
			dw_split = {}
			for k, ds in ds_split.items():
					transformer = module.FeatureTransformer(ds, extractor, **ft_kwargs)
					# df_split[k] = transformer.dataset_feature
					dw_split[k] = transformer.dataset_windows
		else:
			dw_split = utils.split_dataset(dw, splits,
			labels=None if uniform_mode else compactor.label_dict.keys(),
			**sp_kwargs
			)

		return dw_split

