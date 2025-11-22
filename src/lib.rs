use wasm_bindgen::prelude::*;
use candle_core::{Device, Tensor, DType, Var};
use candle_nn::{VarBuilder, VarMap, Optimizer, Module};
use candle_nn::optim::AdamW;

// Inicializa el panic hook para ver errores en la consola del navegador
#[wasm_bindgen]
pub fn init() {
    console_error_panic_hook::set_once();
}

// Definición de un modelo simple (Simulando LoRA layers)
// En un escenario real, aquí definirías la arquitectura completa (Qwen, Llama)
// o solo las capas inyectadas de LoRA.
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
        // Intenta usar WebGPU, si falla usa CPU
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        // Para WebGPU real con candle, se usaría Device::new_wgpu(...) asumiendo soporte async
        // Por simplicidad y compatibilidad WASM sincrónica, usamos CPU aquí como base robusta
        // o WGPU si el binding lo permite directamente.
        
        Ok(TrainingEngine {
            device: Device::Cpu, // Usamos CPU para garantizar compatibilidad WASM inmediata
            varmap: VarMap::new(),
            optimizer: None,
            model: None,
            step_count: 0,
        })
    }

    // Carga los pesos del modelo base (Simulado para este ejemplo)
    pub fn load_model_weights(&self, _weights: &[u8], model_id: String) -> String {
        format!("Model {} metadata loaded. Ready for LoRA init.", model_id)
    }

    // Inicializa las capas LoRA (Entrenables)
    pub fn init_lora(&mut self, rank: usize, _alpha: f64) -> Result<String, JsError> {
        let dim_in = 64;  // Dimensiones simuladas del embedding
        let dim_out = 64; // Dimensiones simuladas de salida

        // Construir el modelo usando VarBuilder asociado a nuestro VarMap
        let vb = VarBuilder::from_varmap(&self.varmap, DType::F32, &self.device);
        
        let layer_a = candle_nn::linear(dim_in, rank, vb.pp("lora_a"))?;
        let layer_b = candle_nn::linear(rank, dim_out, vb.pp("lora_b"))?;

        self.model = Some(LoraModel { layer_a, layer_b });

        // Configurar Optimizador AdamW
        let params = self.varmap.all_vars();
        let opt = AdamW::new(params, candle_nn::ParamsAdamW {
            lr: 0.001, // Learning rate por defecto
            ..Default::default()
        })?;
        
        self.optimizer = Some(opt);

        Ok("LoRA layers initialized and optimizer ready.".to_string())
    }

    // Paso de entrenamiento real: Forward -> Loss -> Backward -> Step
    pub fn train_step(&mut self, input_ids: &[u32], _labels: &[u32]) -> Result<js_sys::Object, JsError> {
        let model = self.model.as_ref().ok_or(JsError::new("Model not initialized"))?;
        let opt = self.optimizer.as_mut().ok_or(JsError::new("Optimizer not initialized"))?;

        // 1. Preparar datos (Simulamos embeddings a partir de input_ids)
        // En producción, usarías una matriz de embeddings real.
        let batch_size = 1;
        let dim = 64;
        // Creamos un tensor aleatorio para simular la entrada del modelo base
        let input_tensor = Tensor::randn(0f32, 1f32, (batch_size, dim), &self.device)?;
        
        // Simulamos "labels" como un objetivo numérico para regresión simple en este ejemplo
        let target_tensor = Tensor::randn(0f32, 1f32, (batch_size, dim), &self.device)?;

        // 2. Forward Pass
        let logits = model.forward(&input_tensor)?;

        // 3. Calcular Loss (MSE para simplicidad en este ejemplo)
        let loss = candle_nn::loss::mse(&logits, &target_tensor)?;

        // 4. Backward Pass (Backpropagation)
        opt.backward_step(&loss)?;

        self.step_count += 1;

        // Retornar resultados a JS
        let loss_val = loss.to_vec0::<f32>()?;
        
        let result = js_sys::Object::new();
        js_sys::Reflect::set(&result, &"loss".into(), &loss_val.into())?;
        js_sys::Reflect::set(&result, &"step".into(), &self.step_count.into())?;

        Ok(result)
    }

    // Exportar los pesos entrenados (Tensores LoRA)
    pub fn export_adapters(&self) -> Result<Vec<u8>, JsError> {
        // Serializar el VarMap a formato safetensors
        // Nota: Esta es una simplificación. Normalmente guardarías solo los tensores entrenados.
        let mut buffer = Vec::new();
        self.varmap.save(&mut buffer)?;
        Ok(buffer)
    }
}
