fn dropout_forward(
    values: &mut [f32],
    p: f32,
    rng: &mut StdRng,
) -> Vec<f32> {
    assert!(p.is_finite() && p >= 0.0 && p < 1.0);

    let mut multipliers = vec![1.0; values.len()];

    if p == 0.0 {
        return multipliers;
    }

    for i in 0..values.len() {
        let keep = rng.random_range(0.0_f32..1.0) >= p;

        let multiplier: f32 = if keep {
            1.0 / (1.0 - p)
        } else {
            0.0
        };

        multipliers[i] = multiplier;
        values[i] *= multiplier;
    }

    multipliers
}

fn dropout_backward(
    gradients: &mut [f32],
    multipliers: &[f32],
) {
    assert_eq!(gradients.len(), multipliers.len());

    for i in 0..gradients.len() {
        gradients[i] *= multipliers[i]
    }
}

pub struct Sample {
    pub input: Vec<f32>,
    pub target: usize,
}

#[derive(Debug)]
pub struct Metrics {
    pub loss: f32,
    pub accuracy: f32,
}

pub fn train_epoch(
    network: &mut Network,
    samples: &[Sample],
    loss: &dyn Loss,
    learning_rate: f32,
    lambda: f32,
) {
    assert!(!samples.is_empty());
    assert!(learning_rate.is_finite() && learning_rate > 0.0);
    assert!(lambda.is_finite() && lambda >= 0.0);

    for sample in samples {
        let pass = network.forward_train(&sample.input);
        let result = loss.evaluate(&pass.output, sample.target);

        let mut gradients =
            network.backward(&pass, &result.gradient);

        network.add_l2_gradients(&mut gradients, lambda);

        network.apply_gradients(&gradients, learning_rate);
    }
}

pub fn evaluate_dataset(
    network: &Network,
    samples: &[Sample],
    loss: &dyn Loss,
) -> Metrics {
    assert!(!samples.is_empty());

    let mut sum: f32 = 0.0;
    let mut correct: usize = 0;

    for sample in samples {
        let pass = network.forward(&sample.input);
        let result = loss.evaluate(&pass.output, sample.target);

        sum += result.value;

        let mut predicted = 0;

        for i in 1..pass.output.len() {
            if pass.output[i] > pass.output[predicted] {
                predicted = i;
            }
        }

        if predicted == sample.target {
            correct += 1;
        }
    }

    let count = samples.len() as f32;

    Metrics {
        loss: sum / count,
        accuracy: correct as f32 / count,
    }
}


#[derive(Debug)]
pub struct LossOutput {
    pub value: f32,
    pub gradient: Vec<f32>,
}

pub trait Loss {
    fn evaluate(&self, logits: &[f32], target: usize) -> LossOutput;
}

pub struct SoftmaxCrossEntropy;

impl Loss for SoftmaxCrossEntropy {
    fn evaluate(&self, logits: &[f32], target: usize) -> LossOutput {
        assert!(logits.len() >= 2);
        assert!(target < logits.len());

        let max = logits
            .iter()
            .copied()
            .max_by(f32::total_cmp)
            .expect("logits не должен быть пустым");

        let exps: Vec<f32> = logits
            .iter()
            .map(|&x| (x - max).exp())
            .collect();

        let sum: f32 = exps.iter().sum();

        let probabilities: Vec<f32> = exps
            .iter()
            .map(|&x| x / sum)
            .collect();

        let loss = sum.ln() - (logits[target] - max);

        let derivative_loss: Vec<f32> = probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                p - if i == target { 1.0 } else { 0.0 }
            })
            .collect();

        LossOutput {
            value: loss,
            gradient: derivative_loss,
        }
    }
}

#[derive(Clone)]
pub struct Network {
    layers: Vec<Dense>,
    dropout: f32,
    rng: StdRng,
}

pub struct ForwardPass {
    pub layer_inputs: Vec<Vec<f32>>,
    pub output: Vec<f32>,
    pub dropout_multipliers: Vec<Option<Vec<f32>>>,
}

impl Network {
    pub fn new(layers: Vec<Dense>) -> Self {
        assert!(layers.len() != 0);
        for i in 1..layers.len(){
            assert_eq!(layers[i - 1].neurons, layers[i].inputs);
        }

        Self {
            layers,
            dropout: 0.0,
            rng: StdRng::seed_from_u64(42),
        }
    }

    pub fn forward_train(&mut self, input: &[f32]) -> ForwardPass {
        let mut current = input.to_vec();
        let mut layer_inputs = Vec::with_capacity(self.layers.len());
        let mut dropout_multipliers = Vec::with_capacity(self.layers.len());

        for (i, layer) in self.layers.iter().enumerate() {
            let mut next = layer.forward(&current);

            let multipliers = if i + 1 < self.layers.len()
                && self.dropout > 0.0
            {
                Some(dropout_forward(
                    &mut next,
                    self.dropout,
                    &mut self.rng,
                ))
            } else {
                None
            };

            layer_inputs.push(current);
            dropout_multipliers.push(multipliers);
            current = next;
        }

        ForwardPass {
            layer_inputs,
            output: current,
            dropout_multipliers,
        }
    }

    pub fn l2_penalty(&self, lambda: f32) -> f32 {
        assert!(lambda.is_finite() && lambda >= 0.0);

        let mut sum = 0.0_f32;

        for layer_index in 0..self.layers.len() {
            let layer = &self.layers[layer_index];

            for weight_index in 0..layer.weights.len() {
                let weight = layer.weights[weight_index];
                sum += weight * weight;
            }
        }

        0.5 * lambda * sum
    }

    pub fn add_l2_gradients( &self, gradients: &mut [DenseGradients], lambda: f32,) {
        assert!(lambda.is_finite() && lambda >= 0.0);
        assert_eq!(gradients.len(), self.layers.len());

        for layer_index in 0..self.layers.len() {
            let layer = &self.layers[layer_index];
            let layer_grad = &mut gradients[layer_index];

            assert_eq!(layer_grad.weights.len(), layer.weights.len());

            for weight_index in 0..layer.weights.len() {
                let weight = layer.weights[weight_index];

                let penalty_gradient: f32 = lambda * weight;
                    

                layer_grad.weights[weight_index] += penalty_gradient;
            }
        }
    }


    pub fn forward(&self, input: &[f32]) -> ForwardPass {
        let mut current = input.to_vec();
        let mut layer_inputs = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let next = layer.forward(&current);
            layer_inputs.push(current);
            current = next;
        }

        ForwardPass {
            layer_inputs,
            output: current,
            dropout_multipliers: vec![None; self.layers.len()],
        }
    }

    pub fn backward(
        &self,
        pass: &ForwardPass,
        grad_output: &[f32],
    ) -> Vec<DenseGradients> {
        assert_eq!(pass.layer_inputs.len(), self.layers.len());
        let mut current_grad = grad_output.to_vec();
        let mut gradients = Vec::with_capacity(self.layers.len());

        assert_eq!(
            pass.dropout_multipliers.len(),
            self.layers.len(),
        );
        for i in (0..self.layers.len()).rev() {
            if let Some(multipliers) = &pass.dropout_multipliers[i] {
                dropout_backward(&mut current_grad, multipliers);
            }

            let layer_gradients = self.layers[i].backward(
                &pass.layer_inputs[i],
                &current_grad,
            );

            current_grad = layer_gradients.input.clone();
            gradients.push(layer_gradients);
        }
        gradients.reverse();
        gradients
    }

    pub fn apply_gradients(
        &mut self,
        gradients: &[DenseGradients],
        learning_rate: f32,
    ) {
        assert_eq!(gradients.len(), self.layers.len());
        for i in 0..self.layers.len() {
            assert_eq!( gradients[i].weights.len(),
                self.layers[i].weights.len());
            assert_eq!( gradients[i].biases.len(),
                self.layers[i].biases.len());
        }
        for i in 0..self.layers.len() {
            self.layers[i].apply_gradients(&gradients[i], learning_rate);
        }
    }
}




fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Длины векторов должны совпадать");
    let mut sum: f32 = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b [i];
    }
    sum
}

fn relu(x: f32) -> f32 {
    let result: f32 = if x > 0.0 {
        x
    } else {
        0.0  
    };
    result
}

#[derive(Clone)]
pub enum Activation {
    Linear,
    Relu,
    Silu,
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

impl Activation {
    fn apply(&self, x: f32) -> f32 {
        match self {
            Self::Linear => x,
            Self::Relu => relu(x),
            Self::Silu => x * sigmoid(x),
        }
    }

    fn derivative(&self, z: f32) -> f32 {
        match self {
            Self::Linear => 1.0,
            Self::Relu => {
                if z > 0.0 { 1.0 } else { 0.0 }
            }
            Self::Silu => {
                let s = sigmoid(z);
                s + z * s * (1.0 - s)
            }
        }
    }
}

#[derive(Clone)]
pub struct Dense {
    inputs: usize,
    neurons: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
    activation: Activation,
}

#[derive(Debug)]
pub struct DenseGradients {
    pub input: Vec<f32>,
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
}

impl Dense {
    pub fn new(
        inputs: usize,
        neurons: usize,
        weights: Vec<f32>,
        biases: Vec<f32>,
        activation: Activation,
    ) -> Self {
        assert!(inputs > 0 && neurons > 0);
        assert_eq!(
            weights.len(),
            inputs * neurons,
            "weights должен иметь длину inputs * neurons"
        );

        assert_eq!(
            biases.len(),
            neurons,
            "biases должен иметь длину neurons"
        );


        Self { inputs, neurons, weights, biases, activation }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(
            input.len(),
            self.inputs,
            "Длина input должна совпадать с inputs"
        );

        let mut outputs = Vec::with_capacity(self.neurons);

        for j in 0..(self.neurons){
            let start = j * self.inputs;
            let end = start + self.inputs;
        
            let neuron_weights = &self.weights[start..end];
            let z: f32 = dot(input, neuron_weights) + self.biases[j];

            let activated = self.activation.apply(z);

            outputs.push(activated);
        }
        outputs
    }
    pub fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
    ) -> DenseGradients {
        assert_eq!(input.len(), self.inputs);
        assert_eq!(grad_output.len(), self.neurons);

        let mut grad_input = vec![0.0_f32; self.inputs];
        let mut grad_weights = vec![0.0_f32; self.weights.len()];
        let mut grad_biases = vec![0.0_f32; self.neurons];

        for j in 0..(self.neurons){
            let start = j * self.inputs;
            let end = start + self.inputs;
        
            let neuron_weights = &self.weights[start..end];
            let z: f32 = dot(input, neuron_weights) + self.biases[j];

            let delta = grad_output[j] * self.activation.derivative(z);

            grad_biases[j] = delta;

            for i in 0..self.inputs {
                let index = j * self.inputs + i;

                grad_weights[index] = delta * input[i];
                grad_input[i] += self.weights[index] * delta;
            }
            
        }

        DenseGradients {
            input: grad_input,
            weights: grad_weights,
            biases: grad_biases,
        }
    }
    pub fn apply_gradients(
        &mut self,
        gradients: &DenseGradients,
        learning_rate: f32,
    ) {
        assert_eq!(gradients.weights.len(), self.weights.len());
        assert_eq!(gradients.biases.len(), self.biases.len());

        for i in 0..self.weights.len() {
            self.weights[i] -= learning_rate * gradients.weights[i];
        }

        for j in 0..self.biases.len() {
            self.biases[j] -= learning_rate * gradients.biases[j];
        }
    }
}


type ModelState = (
    Vec<usize>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<String>,
    f32,
);

impl Network {
    fn export_state(&self) -> ModelState {
        let mut sizes = vec![self.layers[0].inputs];
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut activations = Vec::new();

        for layer in &self.layers {
            sizes.push(layer.neurons);
            weights.push(layer.weights.clone());
            biases.push(layer.biases.clone());
            let name = match layer.activation {
                Activation::Linear => "linear",
                Activation::Relu => "relu",
                Activation::Silu => "silu",
            };
            activations.push(name.to_string());
        }

        (sizes, weights, biases, activations, self.dropout)
    }

    fn from_state(
        sizes: Vec<usize>,
        weights: Vec<Vec<f32>>,
        biases: Vec<Vec<f32>>,
        activations: Vec<String>,
        dropout: f32,
    ) -> Result<Self, String> {
        if sizes.len() < 2 || sizes.contains(&0) || *sizes.last().unwrap() < 2 {
            return Err("Некорректные размеры сети".to_string());
        }
        if !dropout.is_finite() || !(0.0..1.0).contains(&dropout) {
            return Err("dropout должен находиться в диапазоне [0, 1)".to_string());
        }
        let count = sizes.len() - 1;
        if weights.len() != count || biases.len() != count || activations.len() != count {
            return Err("Число наборов параметров не совпадает с числом слоёв".to_string());
        }

        let mut layers = Vec::new();
        for (i, ((w, b), name)) in weights.into_iter()
            .zip(biases).zip(activations).enumerate()
        {
            let expected = sizes[i].checked_mul(sizes[i + 1])
                .ok_or("Слишком большие размеры слоя")?;
            if w.len() != expected || b.len() != sizes[i + 1] {
                return Err(format!("Неверные размеры параметров слоя {i}"));
            }
            if w.iter().chain(&b).any(|value| !value.is_finite()) {
                return Err(format!("NaN или inf в параметрах слоя {i}"));
            }
            let activation = match name.as_str() {
                "linear" => Activation::Linear,
                "relu" => Activation::Relu,
                "silu" => Activation::Silu,
                _ => return Err(format!("Неизвестная активация: {name}")),
            };
            if i == count - 1 && !matches!(activation, Activation::Linear) {
                return Err("Последний слой должен возвращать линейные логиты".to_string());
            }
            layers.push(Dense::new(sizes[i], sizes[i + 1], w, b, activation));
        }

        let mut network = Self::new(layers);
        network.dropout = dropout;
        Ok(network)
    }
}

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Clone)]
#[pyclass(skip_from_py_object)]
struct MLP {
    network: Network,
}
#[pymethods]
impl MLP {
    #[new]
    #[pyo3(signature = (sizes, seed=42, dropout=0.0))]
    fn new(
        sizes: Vec<usize>,
        seed: u64,
        dropout: f32,
    ) -> PyResult<Self> {

        if !dropout.is_finite() || dropout < 0.0 || dropout >= 1.0 {
            return Err(PyValueError::new_err(
                "dropout должен находиться в диапазоне [0, 1)",
            ));
        }

        if sizes.len() < 2 || sizes.contains(&0) {
            return Err(PyValueError::new_err(
                "Нужны размеры входа и слоёв, все больше нуля",
            ));
        }
        if *sizes.last().unwrap() < 2 {
            return Err(PyValueError::new_err(
                "Для классификации нужно минимум два выхода",
            ));
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let mut layers = Vec::new();

        for i in 0..sizes.len() - 1 {
            let inputs = sizes[i];
            let neurons = sizes[i + 1];
            let last = i == sizes.len() - 2;

            let bound = if last {
                (6.0 / (inputs + neurons) as f32).sqrt()
            } else {
                (6.0 / inputs as f32).sqrt()
            };

            let weights = (0..inputs * neurons)
                .map(|_| rng.random_range(-bound..bound))
                .collect();

            let activation = if i == sizes.len() - 2 {
                Activation::Linear
            } else {
                Activation::Relu
            };

            layers.push(Dense::new(
                inputs,
                neurons,
                weights,
                vec![0.0; neurons],
                activation,
            ));
        }

        let mut network = Network::new(layers);
        network.dropout = dropout;
        network.rng = rng;

        Ok(Self { network })
    }

    #[pyo3(signature = (x, y, learning_rate, l2=0.0))]
    fn train_epoch(
        &mut self,
        x: Vec<Vec<f32>>,
        y: Vec<usize>,
        learning_rate: f32,
        l2: f32,
    ) -> PyResult<()> {
        let inputs = self.network.layers[0].inputs;
        let classes = self.network.layers.last().unwrap().neurons;

        if x.is_empty()
            || x.len() != y.len()
            || x.iter().any(|row| row.len() != inputs)
            || y.iter().any(|&label| label >= classes)
        {
            return Err(PyValueError::new_err(
                "Проверь размеры X, y и номера классов",
            ));
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(PyValueError::new_err(
                "learning_rate должен быть конечным и положительным",
            ));
        }

        if !l2.is_finite() || l2 < 0.0 {
            return Err(PyValueError::new_err(
                "l2 должен быть конечным и неотрицательным",
            ));
        }

        let samples: Vec<Sample> = x
            .into_iter()
            .zip(y)
            .map(|(input, target)| Sample { input, target })
            .collect();

        crate::train_epoch(
            &mut self.network,
            &samples,
            &SoftmaxCrossEntropy,
            learning_rate,
            l2,
        );

        Ok(())
    }

    fn predict_logits(&self, x: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let inputs = self.network.layers[0].inputs;

        if x.iter().any(|row| row.len() != inputs) {
            return Err(PyValueError::new_err(
                "Неверное количество входных признаков",
            ));
        }

        Ok(x.iter()
            .map(|row| self.network.forward(row).output)
            .collect())
    }

    fn export_state(&self) -> ModelState {
        self.network.export_state()
    }

    #[staticmethod]
    fn from_state(
        sizes: Vec<usize>,
        weights: Vec<Vec<f32>>,
        biases: Vec<Vec<f32>>,
        activations: Vec<String>,
        dropout: f32,
    ) -> PyResult<Self> {
        let network = Network::from_state(sizes, weights, biases, activations, dropout)
            .map_err(PyValueError::new_err)?;
        Ok(Self { network })
    }

    fn snapshot(&self) -> Self {
        self.clone()
    }
}

#[pymodule]
fn perceptron(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MLP>()
}
