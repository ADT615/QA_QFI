import dimod
import dwave.inspector
import dwave.system

bqm = dimod.generators.ran_r(1, 20)
sampler = dwave.system.EmbeddingComposite(dwave.system.DWaveSampler())
sampleset = sampler.sample(bqm, num_reads=100)
dwave.inspector.show(sampleset)