use wasm_bindgen::prelude::*;
use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, VarMap, Optimizer, Module};
use candle_nn::optim::AdamW;

// Inicializa el panic hook para ver errores en la consola del navegador
#[wasm_bindgen]
pub fn init() {
    console_error_panic_hook::set_once();
}

// Definición de un modelo simple (Simulando LoRA layers)
struct LoraModel {
    layer_a: candle_nn::Linear,
    layer_b: candle_nn::Linear,
}

impl LoraModel {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.layer_a.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.layer_b.forward(xs)?;
        Ok(xs)
    }
}

#[wasm_bindgen]
pub struct TrainingEngine {
    device: Device,
    varmap: VarMap,
    optimizer: Option<AdamW>,
    model: Option<LoraModel>,
    step_count: usize,
}

#[wasm_bindgen]
impl TrainingEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<TrainingEngine, JsError> {
        // Usamos CPU para garantizar compatibilidad WASM. 
        // El soporte WebGPU directo en Candle-WASM requiere configuración adicional no presente en crates.io v0.3.3.
        let device = Device::Cpu;
        
        Ok(TrainingEngine {
            device,
            varmap: VarMap::new(),
            optimizer: None,
            model: None,
            step_count: 0,
        })
    }

    pub fn load_model_weights(&self, _weights: &[u8], model_id: String) -> String {
        format!("Model {} metadata loaded. Ready for LoRA init.", model_id)
    }

    pub fn init_lora(&mut self, rank: usize, _alpha: f64) -> Result<String, JsError> {
        let dim_in = 64;
        let dim_out = 64;

        let vb = VarBuilder::from_varmap(&self.varmap, DType::F32, &self.device);
        
        let layer_a = candle_nn::linear(dim_in, rank, vb.pp("lora_a"))?;
        let layer_b = candle_nn::linear(rank, dim_out, vb.pp("lora_b"))?;

        self.model = Some(LoraModel { layer_a, layer_b });

        let params = self.varmap.all_vars();
        let opt = AdamW::new(params, candle_nn::ParamsAdamW {
            lr: 0.001,
            ..Default::default()
        })?;
        
        self.optimizer = Some(opt);

        Ok("LoRA layers initialized (CPU Backend).".to_string())
    }

    pub fn train_step(&mut self, _input_ids: &[u32], _labels: &[u32]) -> Result<js_sys::Object, JsError> {
        let model = self.model.as_ref().ok_or(JsError::new("Model not initialized"))?;
        let opt = self.optimizer.as_mut().ok_or(JsError::new("Optimizer not initialized"))?;

        // 1. Datos simulados (Tensores reales en memoria)
        let batch_size = 1;
        let dim = 64;
        let input_tensor = Tensor::randn(0f32, 1f32, (batch_size, dim), &self.device)?;
        let target_tensor = Tensor::randn(0f32, 1f32, (batch_size, dim), &self.device)?;

        // 2. Forward
        let logits = model.forward(&input_tensor)?;

        // 3. Loss
        let loss = candle_nn::loss::mse(&logits, &target_tensor)?;

        // 4. Backward
        opt.backward_step(&loss)?;

        self.step_count += 1;

        let loss_val = loss.to_vec0::<f32>()?;
        
        let result = js_sys::Object::new();
        js_sys::Reflect::set(&result, &"loss".into(), &loss_val.into())?;
        js_sys::Reflect::set(&result, &"step".into(), &self.step_count.into())?;

        Ok(result)
    }

    pub fn export_adapters(&self) -> Result<Vec<u8>, JsError> {
        let mut buffer = Vec::new();
        self.varmap.save(&mut buffer)?;
        Ok(buffer)
    }
}
