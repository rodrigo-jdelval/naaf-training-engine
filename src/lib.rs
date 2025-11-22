use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn init_engine() -> String {
    "Motor WASM iniciado correctamente desde la Nube!".to_string()
}

#[wasm_bindgen]
pub fn train_step(loss: f64) -> f64 {
    // Simulación simple: reduce el loss un 10%
    loss * 0.90
}
