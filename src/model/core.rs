use rust_bert::{
    gpt_neo::{
        GptNeoConfigResources, GptNeoMergesResources, GptNeoModelResources, GptNeoVocabResources,
    },
    pipelines::{
        common::ModelType,
        text_generation::{TextGenerationConfig, TextGenerationModel},
    },
    resources::RemoteResource,
};

pub fn init_model() -> TextGenerationModel {
    // Retrieve model resource
    let model_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoModelResources::GPT_NEO_2_7B,
    ));

    // Retrieve config resource
    let config_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoConfigResources::GPT_NEO_2_7B,
    ));

    // Retrieve vocab resource
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoVocabResources::GPT_NEO_2_7B,
    ));

    // Retrieve merges resource
    let merges_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoMergesResources::GPT_NEO_2_7B,
    ));

    // Assemble configuration
    let generate_config = TextGenerationConfig {
        model_type: ModelType::GPTNeo,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        num_beams: 5,
        no_repeat_ngram_size: 2,
        max_length: 100,
        ..Default::default()
    };

    // Return model
    TextGenerationModel::new(generate_config).unwrap()
}
