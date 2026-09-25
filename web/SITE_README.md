# Fly Terrarium — the site

**[Open it](https://s31fdev.github.io/fly-terrarium/)** · [source code](https://github.com/s31fdev/fly-terrarium/tree/main)

A fly in a Petri dish, driven by a model of its whole nervous system built from the connectome:
a female from FlyWire (138,639 neurons) or a male from the MaleCNS (165,122 neurons). The model is the
leaky integrate-and-fire model of Shiu et al. (Nature, 2024), running right in the browser. Sugar,
bitter, a puff of air and a shadow reach real sensory neurons, and the fly does what the spikes of its
descending neurons tell it. The page says what is real here and what is made up.

This branch is the built static site: HTML, JavaScript and the compressed connectome data. It is
published from the `main` branch by `publish_web.sh`.

Code: [MIT license](LICENSE). Data: FlyWire v783 (Dorkenwald et al. 2024, Schlegel et al. 2024), license CC BY-NC 4.0,
non-commercial use only. MaleCNS v1.0 (FlyEM / HHMI Janelia, University of Cambridge, MRC LMB,
Google Research; Berg et al.), license CC BY 4.0. The data are repacked and compressed, neuron
coordinates rotated, male synapses weakened 1.8-fold.
