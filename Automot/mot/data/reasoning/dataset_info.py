
from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset_fast_thinking import SftJSONLIterableDataset
from .vlm_dataset_senna import SftJSONLIterableDatasetSenna
from .vlm_dataset_fast_thinking_pdmlite import SftJSONLIterableDatasetpdmtraj
DATASET_REGISTRY = {
    # 't2i_pretrain': T2IIterableDataset,
    'motvla_slow': SftJSONLIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'vlm_sft_senna': SftJSONLIterableDatasetSenna,
    'vlm_sft_pdmtraj': SftJSONLIterableDatasetpdmtraj,
    # 'unified_edit': UnifiedEditIterableDataset,
}


DATASET_INFO = {
    # 't2i_pretrain': {
    #     't2i': {
    #         'data_dir': 'your_data_path/bagel_example/t2i', # path of the parquet files
    #         'num_files': 10, # number of data units to be sharded across all ranks and workers
    #         'num_total_samples': 1000, # number of total samples in the dataset
    #     },
    # },
    # 'unified_edit':{
    #     'seedxedit_multi': {
    #         'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
    #         'num_files': 10,
    #         'num_total_samples': 1000,
    #         "parquet_info_path": 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
	# 	},
    # }
    # 'vlm_sft_senna': {
    #     'nuscenes_senna': {
    #         'data_dir': '/mnt/data2/nuscenes',
	# 		'jsonl_path': '/share-data/senna_nusc_output/senna_bagel.jsonl',
	# 		'num_total_samples': 23930
	# 	},
    # },
    'vlm_sft_pdmtraj': {
		'pdm_lite_traj': {
            'data_dir': '/share-data/pdm_lite',
			'jsonl_path': '/root/qihang_projects/AutoMoTive/data/pdm_lite_jsonl/pdm_train.jsonl',
			'num_total_samples': 605419
		},
		'pdm_lite_traj_eval': {
            'data_dir': '/share-data/pdm_lite',
			'jsonl_path': '/root/qihang_projects/AutoMoTive/data/pdm_lite_jsonl/pdm_val.jsonl',
			'num_total_samples': 10560
		},
	},
    'vlm_sft_senna': {
        'nuscenes_senna': {
            'data_dir': '/mnt/data2/nuscenes',
			'jsonl_path': '/share-data/senna_nusc_output_soft_label/senna_nusc_train_bagel.jsonl',
			'num_total_samples': 23930
		},
        'nuscenes_count': {
            'data_dir': '/mnt/data2/nuscenes',
			'jsonl_path': '/root/qihang_projects/AutoMoTive/data/pdm_lite_jsonl/nuscenes_count_train_mot.jsonl',
			'num_total_samples': 11555
		},
        'pdm_lite_meta_action': {
			'data_dir': '/mnt/data2/pdm_lite',
			'jsonl_path': '/root/qihang_projects/AutoMoTive/data/pdm_lite_jsonl/pdm_meta_action_train_balanced_aligned.jsonl',
			'num_total_samples': 14967
		},
    },
    'vlm_sft': {
        'pdm_lite_fast_thinking': {
			'data_dir': '/mnt/data2/pdm_lite',
			'jsonl_path': '/root/qihang_projects/dev-oscar-qwen3vl-transformers-template-po-qihang/data/pdm_lite_jsonl/pdm_multiframes_train_balanced.jsonl',
			'num_total_samples': 16384
		},
        'pdm_lite_fast_slow_thinking': {
			'data_dir': '/mnt/data2/pdm_lite',
			'jsonl_path': '/root/qihang_projects/dev-oscar-qwen3vl-transformers-template-po-qihang/data/pdm_lite_jsonl/pdm_multiframes_train_aug.jsonl',
			'num_total_samples': 64868
		},
        'llava_ov': {
			'data_dir': '/root/qihang_projects/BAGEL/data/bagel_example/vlm/images',
			'jsonl_path': '/root/qihang_projects/BAGEL/data/bagel_example/vlm/llava_ov_1000_slow_reasoning.jsonl',
			'num_total_samples': 1000
		},
        'lingoqa': {
			'data_dir': '/root/oscar_projects',
			'jsonl_path': '/root/oscar_projects/data/LingoQA/jsonl/lingoqa_train.jsonl',
			'num_total_samples': 40012
		},
        'lingoqa_4.6k': {
			'data_dir': '/root/oscar_projects',
			'jsonl_path': '/root/oscar_projects/data/LingoQA/training_data_4_6k.jsonl',
			'num_total_samples': 4600
		},
        'coco': {
			'data_dir': '/mnt',
			'jsonl_path': '/mnt/data2/COCO/jsonl/coco.jsonl',
			'num_total_samples': 902960
		},
        'iconqa': {
			'data_dir': '/mnt',
			'jsonl_path': '/mnt/data2/IConQA/iconqa.jsonl',
			'num_total_samples': 107439
		},
        'scienceqa': {
			'data_dir': '/mnt',
			'jsonl_path': '/mnt/data2/ScienceQA/scienceqa.jsonl',
			'num_total_samples': 6218
		},
       'textvqa': {
			'data_dir': '/mnt',
			'jsonl_path': '/mnt/data2/textvqa/textVQA.jsonl',
			'num_total_samples': 56555
		},
       'sbu': {
			'data_dir': '/mnt',
			'jsonl_path': '/mnt/data2/sbu_images/sbu_filtered.jsonl',
			'num_total_samples': 68670
		},
    },
    'motvla_slow': {
        'lingoqa': {
			'data_dir': '/root/oscar_projects',
			'jsonl_path': '/root/oscar_projects/data/LingoQA/jsonl/lingoqa_train.jsonl',
			'num_total_samples': 40012
		},
        'lingoqa_4.6k': {
			'data_dir': '/root/oscar_projects',
			'jsonl_path': '/root/oscar_projects/data/LingoQA/training_data_4_6k.jsonl',
			'num_total_samples': 4600
		},
        'drama': {
			'data_dir': '/root/oscar_projects',
			'jsonl_path': '/root/oscar_projects/data/DRAMA/jsonl/drama_train.jsonl',
			'num_total_samples': 33842
		},
        'coda-explanation': {
			'data_dir': '/root/oscar_projects',
			'jsonl_path': '/root/oscar_projects/data/CODA/jsonl/explanation_train.jsonl',
			'num_total_samples': 10227
		},
        'coda-grounding': {
			'data_dir': '/root/oscar_projects',
			'jsonl_path': '/root/oscar_projects/data/CODA/jsonl/grounding_train.jsonl',
			'num_total_samples': 4384
		},
        'coda-perception': {
			'data_dir': '/root/oscar_projects',
			'jsonl_path': '/root/oscar_projects/data/CODA/jsonl/perception_train.jsonl',
			'num_total_samples': 4384
		},
        'coda-suggestion': {
			'data_dir': '/root/oscar_projects',
			'jsonl_path': '/root/oscar_projects/data/CODA/jsonl/suggestion_train.jsonl',
			'num_total_samples': 4384
		},
        'b2d-action': {
			'data_dir': '/root/oscar_projects',
			'jsonl_path': '/root/data/b2d_vlm/b2d_action_train.jsonl',
			'num_total_samples': 4968
		},
        'b2d-scenary': {
			'data_dir': '/root/oscar_projects',
			'jsonl_path': '/root/data/b2d_vlm/b2d_scenary_train.jsonl',
			'num_total_samples': 1498
		},
        'omnidrive-action': {
			'data_dir': '/root/oscar_projects',
			'jsonl_path': '/root/oscar_projects/data/omnidrive/action_train.jsonl',
			'num_total_samples': 27959
		},
        'omnidrive-perception': {
			'data_dir': '/root/oscar_projects',
			'jsonl_path': '/root/oscar_projects/data/omnidrive/perception_train.jsonl',
			'num_total_samples': 259981
		},
    },
}