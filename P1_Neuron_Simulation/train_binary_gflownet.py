import torch.cuda as cuda

from src.binary_engine import GFlowBinaryEngine
from src.dataset import NeuronDataset
from src.model import FlowModel
from src.simulators import NetworkSystemSimulator


def main() -> None:
    n_neurons = 3
    model = FlowModel(n_neurons=n_neurons, num_hid=512)
    simulator = NetworkSystemSimulator(device=("cuda" if cuda.is_available() else "cpu"))
    engine = GFlowBinaryEngine(model, simulator)
    train_data = NeuronDataset("Id_3x3.pt", map_location=engine.device)

    engine.fit(train_data, n_epochs=10, batch_size=50)


if __name__ == "__main__":
    main()