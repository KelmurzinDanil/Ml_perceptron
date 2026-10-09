use perceptron::{
    Activation, Dense, Network, Sample, SoftmaxCrossEntropy,
    evaluate_dataset, train_epoch,
};

fn main() {
    let samples = vec![
        Sample { input: vec![-2.0, -1.0], target: 0 },
        Sample { input: vec![-1.0, -2.0], target: 0 },
        Sample { input: vec![1.0, 2.0], target: 1 },
        Sample { input: vec![2.0, 1.0], target: 1 },
    ];

    let hidden = Dense::new(
        2,
        3,
        vec![
             0.2, -0.1,
            -0.3,  0.4,
             0.1,  0.2,
        ],
        vec![0.1, 0.1, 0.1],
        Activation::Relu,
    );

    let output = Dense::new(
        3,
        2,
        vec![
             0.2, -0.1,  0.3,
            -0.2,  0.1, -0.3,
        ],
        vec![0.0, 0.0],
        Activation::Linear,
    );

    let mut network = Network::new(vec![hidden, output]);
    let loss = SoftmaxCrossEntropy;

    println!(
        "Before: {:?}",
        evaluate_dataset(&network, &samples, &loss),
    );

    for epoch in 1..=500 {
        train_epoch(&mut network, &samples, &loss, 0.05);

        if epoch % 50 == 0 {
            let metrics = evaluate_dataset(&network, &samples, &loss);

            println!(
                "Epoch {epoch}: loss={:.6}, accuracy={:.1}%",
                metrics.loss,
                metrics.accuracy * 100.0,
            );
        }
    }
}