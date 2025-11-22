use wasm_bindgen::prelude::*;

// Inicializador que llamaremos desde JS para confirmar que cargó
#[wasm_bindgen]
pub fn init_engine() -> String {
    "Motor WASM de NAAF cargado correctamente!".to_string()
}

// Función de entrenamiento simulada (para probar)
#[wasm_bindgen]
pub fn train_step_simulated(current_loss: f64) -> f64 {
    // Reduce el loss un 10% cada paso
    let new_loss = current_loss * 0.90;
    if new_loss < 0.001 { 0.0 } else { new_loss }
}
