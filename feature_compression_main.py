import argparse
from feature_compression.compress_features import run_compression
# from feature_compression.train_autoencoder import fit_autoencoder

def generate_and_compress(activations_dir,model_type: str, filter_config: dict,
                          trafo_type: str, num_components: list):

    save_dir = (f'E:/LLM/Algonauts_2023/compress_features/{model_type}/' +
                f'{filter_config["filter_name"]}/' )
    feature_dir = activations_dir + f'/{model_type}'
    for num_comps in num_components:
        run_compression(feature_dir, save_dir, model_type, trafo_type, num_comps)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_type', '--model_type',
                        default='InterVideo/InternVideo-MM-B-16-768',#default='Marlin/marlin_vit_small_ytf_384',
                        help='Model to generate activations.')
    parser.add_argument('-trafo_type', '--trafo_type',
                        default='pca', help='Compression Technique: umap,autoencoder,pca')
    parser.add_argument('-filter_name', '--filter_name',
                        default='mean', help='Filter Technique.')
    parser.add_argument('-num_components', '--num_components',
                        default=[50, 60, 70, 80, 90, 100], type=list, help='Dimension to reduce to.')#default=[50, 60, 70, 80, 90, 100], type=list, help='Dimension to reduce to.')
    parser.add_argument('-activations_dir', '--activations_dir',
                        default="E:/LLM/Algonauts_2023/features", help='Dimension to reduce to.')
    cmd_args = parser.parse_args()

    filter_config = {"filter_name": cmd_args.filter_name}
    generate_and_compress(cmd_args.activations_dir,cmd_args.model_type, filter_config,
                          cmd_args.trafo_type, cmd_args.num_components)

