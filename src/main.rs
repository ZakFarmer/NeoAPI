pub mod model;

use model::core::init_model;
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

fn main() {
    // Initialise model
    let model = init_model();

    loop {
        let mut line = String::new();

        std::io::stdin().read_line(&mut line).unwrap();

        let split = line.split('/').collect::<Vec<&str>>();

        let slc = split.as_slice();

        let output = model.generate(&slc[1..], Some(slc[0]));

        for sentence in output {
            println!("{}", sentence);
        }
    }
}
