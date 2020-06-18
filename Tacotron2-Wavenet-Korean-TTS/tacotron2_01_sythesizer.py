from synthesizer import Synthesizer

synthesizer = Synthesizer()
synthesizer.load(checkpoint_path='logdir-tacotron2/moon+son+kss_2020-05-12_02-13-33',
                 num_speakers=3,
                 checkpoint_step=None,
                 inference_prenet_dropout=False)

f = open('gen/Inputs/test.txt', mode='rt', encoding='utf-8')
text = f.readline()

audio = synthesizer.synthesize(texts=text,
                               base_path="gen/Outputs",
                               speaker_ids=[2], 
                               attention_trim=True, 
                               base_alignment_path=None, 
                               isKorean=True)[0]

