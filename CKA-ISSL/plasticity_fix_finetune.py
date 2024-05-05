import numpy as np
import matplotlib.pyplot as plt

joint = [[1.0] * 9] * 9

fix = [[0.99770314, 0.97815824, 0.97815824, 0.9820196, 0.9930763, 0.9930763, 0.99040073, 0.9864058, 0.9820196,
        0.9064485, 0.950826, 0.950826, 0.91496277, 0.8955792, 0.9064485, 0.8744784, 0.8490198, 0.8490198, 0.8403045,
        0.85245323, 0.8744784, 0.8055212, 0.8149314, 0.8149314, 0.8548404, 0.85254794, 0.8055212, 0.71988684, 0.7343633,
        0.7343633, 0.67415357, 0.68194896, 0.71988684, 0.68487346, 0.7026244, 0.7026244, 0.72046095, 0.71650237,
        0.68487346, 0.58024246, 0.5815882, 0.5815882, 0.6083747, 0.60687625, 0.58024246, 0.46870872, 0.4676761,
        0.4676761, 0.5235659, 0.52155, 0.46870872],
       [0.99763954, 0.98626894, 0.98626894, 0.9831154, 0.99241954, 0.99241954, 0.9914701, 0.98703194, 0.9831154,
        0.8919045, 0.9475615, 0.9475615, 0.90461814, 0.88380075, 0.8919045, 0.8609498, 0.83137137, 0.83137137,
        0.83002293, 0.8346703, 0.8609498, 0.7752965, 0.7889293, 0.7889293, 0.81763244, 0.81429464, 0.7752965,
        0.69793594, 0.71006703, 0.71006703, 0.621739, 0.6327409, 0.69793594, 0.6648508, 0.68079656, 0.68079656,
        0.7228257, 0.71690845, 0.6648508, 0.58548486, 0.5702767, 0.5702767, 0.5938263, 0.589765, 0.58548486, 0.46262038,
        0.47141227, 0.47141227, 0.48806134, 0.49075454, 0.46262038],
       [0.99734604, 0.97961575, 0.97961575, 0.97555023, 0.9906139, 0.9906139, 0.9883071, 0.9831537, 0.97555023,
        0.8993082, 0.9362435, 0.9362435, 0.9019719, 0.8929005, 0.8993082, 0.87285197, 0.8566437, 0.8566437, 0.86205596,
        0.8732546, 0.87285197, 0.7759341, 0.7941092, 0.7941092, 0.8382669, 0.8381823, 0.7759341, 0.70014834, 0.712972,
        0.712972, 0.6587102, 0.6651863, 0.70014834, 0.65414304, 0.6799661, 0.6799661, 0.65490246, 0.6515381, 0.65414304,
        0.5637778, 0.54208994, 0.54208994, 0.49927682, 0.4974378, 0.5637778, 0.47066662, 0.447984, 0.447984, 0.4014037,
        0.40164644, 0.47066662],
       [0.9988441, 0.9890137, 0.9890137, 0.9873754, 0.9941345, 0.9941345, 0.992565, 0.98836815, 0.9873754, 0.8992043,
        0.94549775, 0.94549775, 0.89571947, 0.8899879, 0.8992043, 0.88556933, 0.8648207, 0.8648207, 0.85648704,
        0.8677857, 0.88556933, 0.80506206, 0.83085674, 0.83085674, 0.84835887, 0.84721684, 0.80506206, 0.7232498,
        0.7334787, 0.7334787, 0.66051805, 0.6664683, 0.7232498, 0.698317, 0.7175694, 0.7175694, 0.73534536, 0.731646,
        0.698317, 0.6155982, 0.61296135, 0.61296135, 0.6128263, 0.60976565, 0.6155982, 0.49993387, 0.49407625,
        0.49407625, 0.46382043, 0.46577477, 0.49993387],
       [0.9976348, 0.9761607, 0.9761607, 0.9772748, 0.98969567, 0.98969567, 0.9859762, 0.97988683, 0.9772748,
        0.90289843, 0.95540917, 0.95540917, 0.93531346, 0.89566314, 0.90289843, 0.88247865, 0.8556685, 0.8556685,
        0.8660812, 0.8766631, 0.88247865, 0.75498366, 0.78391457, 0.78391457, 0.84144723, 0.84330076, 0.75498366,
        0.66838336, 0.67915833, 0.67915833, 0.65709907, 0.6648796, 0.66838336, 0.670378, 0.6888406, 0.6888406,
        0.7279434, 0.72389364, 0.670378, 0.5780869, 0.56857175, 0.56857175, 0.59725386, 0.5945913, 0.5780869, 0.4754549,
        0.4555327, 0.4555327, 0.49277723, 0.49256763, 0.4754549],
       [0.99749583, 0.9792661, 0.9792661, 0.9749812, 0.99065393, 0.99065393, 0.9856078, 0.9811956, 0.9749812,
        0.88195205, 0.9435347, 0.9435347, 0.91448426, 0.88600826, 0.88195205, 0.8612932, 0.83420783, 0.83420783,
        0.8504408, 0.8641235, 0.8612932, 0.7458171, 0.7642695, 0.7642695, 0.8373213, 0.8353624, 0.7458171, 0.6675705,
        0.6815136, 0.6815136, 0.64389265, 0.6485678, 0.6675705, 0.6408318, 0.65931815, 0.65931815, 0.7023649,
        0.69735974, 0.6408318, 0.5476157, 0.5421599, 0.5421599, 0.5830206, 0.5800064, 0.5476157, 0.4332694, 0.4396015,
        0.4396015, 0.48642334, 0.48622552, 0.4332694],
       [0.99646854, 0.9754802, 0.9754802, 0.9722782, 0.9894512, 0.9894512, 0.9856374, 0.98017627, 0.9722782, 0.8835688,
        0.9440792, 0.9440792, 0.9132888, 0.8667483, 0.8835688, 0.8530918, 0.8276573, 0.8276573, 0.84407413, 0.8519176,
        0.8530918, 0.73308724, 0.75129986, 0.75129986, 0.81338036, 0.80797464, 0.73308724, 0.65797895, 0.6765529,
        0.6765529, 0.6494111, 0.65243846, 0.65797895, 0.66398346, 0.67217654, 0.67217654, 0.74735165, 0.7389727,
        0.66398346, 0.56789804, 0.56181175, 0.56181175, 0.6296971, 0.6289822, 0.56789804, 0.40179685, 0.43339217,
        0.43339217, 0.50843954, 0.50730866, 0.40179685],
       [0.99776316, 0.9823698, 0.9823698, 0.9796173, 0.99201274, 0.99201274, 0.98936534, 0.9846463, 0.9796173,
        0.88724124, 0.94090503, 0.94090503, 0.8917182, 0.84035313, 0.88724124, 0.8444065, 0.8150243, 0.8150243,
        0.81816167, 0.8198015, 0.8444065, 0.74316657, 0.7507374, 0.7507374, 0.78549445, 0.78546596, 0.74316657,
        0.63943636, 0.65291536, 0.65291536, 0.60479003, 0.60651904, 0.63943636, 0.6195373, 0.6407444, 0.6407444,
        0.706905, 0.69708544, 0.6195373, 0.53596634, 0.52768564, 0.52768564, 0.61293656, 0.6090018, 0.53596634,
        0.43157372, 0.3999936, 0.3999936, 0.4388333, 0.43906632, 0.43157372],
       [0.99843353, 0.9887627, 0.9887627, 0.9873607, 0.99352145, 0.99352145, 0.9930466, 0.9892238, 0.9873607,
        0.90590125, 0.9554564, 0.9554564, 0.92396843, 0.9084288, 0.90590125, 0.8799638, 0.85061353, 0.85061353,
        0.8634603, 0.87541294, 0.8799638, 0.7799103, 0.7858911, 0.7858911, 0.8579487, 0.8618194, 0.7799103, 0.6927538,
        0.70890516, 0.70890516, 0.6603225, 0.66456914, 0.6927538, 0.68634814, 0.7030003, 0.7030003, 0.7509893,
        0.7442585, 0.68634814, 0.617664, 0.61735225, 0.61735225, 0.6650859, 0.6634825, 0.617664, 0.5239694, 0.4843012,
        0.4843012, 0.49422035, 0.49499008, 0.5239694],
       [0.99819756, 0.98813033, 0.98813033, 0.9885599, 0.9930359, 0.9930359, 0.9934036, 0.9887286, 0.9885599, 0.8993528,
        0.9492898, 0.9492898, 0.9198889, 0.9059652, 0.8993528, 0.87252986, 0.8484644, 0.8484644, 0.8588293, 0.871933,
        0.87252986, 0.78077066, 0.7870564, 0.7870564, 0.85147476, 0.84966516, 0.78077066, 0.70242333, 0.719317,
        0.719317, 0.6696304, 0.67733943, 0.70242333, 0.6894094, 0.70608795, 0.70608795, 0.7386451, 0.73491246,
        0.6894094, 0.6131394, 0.60965484, 0.60965484, 0.67539793, 0.6712371, 0.6131394, 0.60091156, 0.53182715,
        0.53182715, 0.5594326, 0.5628745, 0.60091156]]

finetune = [
    [0.99989605, 0.99846435, 0.99846435, 0.9968591, 0.9982876, 0.9982876, 0.99733, 0.99605566, 0.9968591, 0.9699875,
     0.9775733, 0.9775733, 0.9812978, 0.95280445, 0.9699875, 0.92864317, 0.92040575, 0.92040575, 0.9566857, 0.9527395,
     0.92864317, 0.89104915, 0.8861485, 0.8861485, 0.9152885, 0.9088594, 0.89104915, 0.85076386, 0.86345655, 0.86345655,
     0.85554105, 0.86428154, 0.85076386, 0.78835976, 0.7995087, 0.7995087, 0.61593467, 0.6290426, 0.78835976, 0.6922888,
     0.69044864, 0.69044864, 0.5434816, 0.5395828, 0.6922888, 0.49455968, 0.55316186, 0.55316186, 0.0, 0.0, 0.49455968],
    [0.9998889, 0.99843407, 0.99843407, 0.9973733, 0.99782455, 0.99782455, 0.9973658, 0.9957127, 0.9973733, 0.96440446,
     0.97597843, 0.97597843, 0.9754874, 0.9292382, 0.96440446, 0.9266325, 0.91334534, 0.91334534, 0.9539224, 0.9466692,
     0.9266325, 0.87699515, 0.86669123, 0.86669123, 0.88694346, 0.8768936, 0.87699515, 0.8201479, 0.8299298, 0.8299298,
     0.82453567, 0.8346067, 0.8201479, 0.7484455, 0.76283, 0.76283, 0.5471919, 0.57203925, 0.7484455, 0.6508385,
     0.6455719, 0.6455719, 0.49662173, 0.49355546, 0.6508385, 0.46059203, 0.5803319, 0.5803319, 0.0, 0.0, 0.46059203],
    [0.99987274, 0.99795085, 0.99795085, 0.9960665, 0.99740845, 0.99740845, 0.99637127, 0.9948145, 0.9960665, 0.9689497,
     0.97279906, 0.97279906, 0.9771942, 0.95418143, 0.9689497, 0.9333781, 0.9341407, 0.9341407, 0.9664983, 0.96314275,
     0.9333781, 0.89382344, 0.8754977, 0.8754977, 0.8911882, 0.88701034, 0.89382344, 0.85151803, 0.8670647, 0.8670647,
     0.8522254, 0.86107415, 0.85151803, 0.75999486, 0.77985626, 0.77985626, 0.56868684, 0.5820077, 0.75999486,
     0.6671512, 0.6448238, 0.6448238, 0.41809648, 0.41442922, 0.6671512, 0.51653147, 0.5777809, 0.5777809, 0.0, 0.0,
     0.51653147],
    [0.99992853, 0.9990569, 0.9990569, 0.9981998, 0.9984165, 0.9984165, 0.9982697, 0.9971125, 0.9981998, 0.97374415,
     0.9791954, 0.9791954, 0.97625977, 0.95046735, 0.97374415, 0.94083434, 0.94497347, 0.94497347, 0.9686094, 0.9644933,
     0.94083434, 0.914796, 0.9005614, 0.9005614, 0.9012746, 0.8947712, 0.914796, 0.8717837, 0.88919353, 0.88919353,
     0.85338855, 0.8633052, 0.8717837, 0.80428874, 0.8179439, 0.8179439, 0.6682253, 0.67553455, 0.80428874, 0.6987309,
     0.69555134, 0.69555134, 0.5626953, 0.5576321, 0.6987309, 0.54127574, 0.59314686, 0.59314686, 0.0, 0.0, 0.54127574],
    [0.99988747, 0.9983914, 0.9983914, 0.99567044, 0.9970778, 0.9970778, 0.9955957, 0.9932189, 0.99567044, 0.96799624,
     0.9794791, 0.9794791, 0.984459, 0.9559864, 0.96799624, 0.9403815, 0.9350267, 0.9350267, 0.96514165, 0.9638473,
     0.9403815, 0.90026844, 0.8941626, 0.8941626, 0.8935354, 0.8921155, 0.90026844, 0.86414856, 0.87876296, 0.87876296,
     0.85520136, 0.86496055, 0.86414856, 0.8112784, 0.8236978, 0.8236978, 0.6478144, 0.6588505, 0.8112784, 0.6813831,
     0.66732824, 0.66732824, 0.5177623, 0.5138719, 0.6813831, 0.5300273, 0.58753914, 0.58753914, 0.0, 0.0, 0.5300273],
    [0.9998904, 0.99858874, 0.99858874, 0.99576354, 0.99710435, 0.99710435, 0.99547106, 0.99356246, 0.99576354,
     0.96626806, 0.9776886, 0.9776886, 0.98116523, 0.94631284, 0.96626806, 0.93592846, 0.9318377, 0.9318377, 0.96617085,
     0.9643672, 0.93592846, 0.8830895, 0.8627383, 0.8627383, 0.90011746, 0.8934517, 0.8830895, 0.83175504, 0.84610367,
     0.84610367, 0.8558319, 0.86401576, 0.83175504, 0.7677017, 0.7832631, 0.7832631, 0.59432185, 0.61114097, 0.7677017,
     0.6539301, 0.6419421, 0.6419421, 0.4838536, 0.47994938, 0.6539301, 0.49273452, 0.5588848, 0.5588848, 0.0, 0.0,
     0.49273452],
    [0.99987024, 0.99848497, 0.99848497, 0.9947562, 0.9959146, 0.9959146, 0.99379444, 0.991048, 0.9947562, 0.95919126,
     0.9730562, 0.9730562, 0.9751091, 0.933402, 0.95919126, 0.9260505, 0.92097837, 0.92097837, 0.95714307, 0.95446515,
     0.9260505, 0.86573, 0.8486809, 0.8486809, 0.87957066, 0.872155, 0.86573, 0.811993, 0.823618, 0.823618, 0.8282107,
     0.8353785, 0.811993, 0.7209506, 0.74381775, 0.74381775, 0.57789433, 0.5877796, 0.7209506, 0.6305092, 0.6146519,
     0.6146519, 0.47014186, 0.469548, 0.6305092, 0.43840134, 0.54385763, 0.54385763, 0.0, 0.0, 0.43840134],
    [0.9998992, 0.9984971, 0.9984971, 0.99630773, 0.9964634, 0.9964634, 0.9951717, 0.9920672, 0.99630773, 0.969099,
     0.9769259, 0.9769259, 0.9762742, 0.939556, 0.969099, 0.92618614, 0.92056423, 0.92056423, 0.95167464, 0.9449267,
     0.92618614, 0.8740796, 0.8674711, 0.8674711, 0.8808267, 0.8754169, 0.8740796, 0.8077609, 0.8216396, 0.8216396,
     0.78604877, 0.79622966, 0.8077609, 0.68132323, 0.6978951, 0.6978951, 0.5216029, 0.5350743, 0.68132323, 0.5817043,
     0.5773069, 0.5773069, 0.4756917, 0.47146738, 0.5817043, 0.46612975, 0.5052895, 0.5052895, 0.0, 0.0, 0.46612975],
    [0.9998968, 0.9986614, 0.9986614, 0.9982037, 0.9986857, 0.9986857, 0.99850726, 0.99739665, 0.9982037, 0.9735756,
     0.9787262, 0.9787262, 0.98237354, 0.9539341, 0.9735756, 0.93986225, 0.93875533, 0.93875533, 0.966798, 0.9653573,
     0.93986225, 0.902029, 0.8813993, 0.8813993, 0.9018306, 0.90149564, 0.902029, 0.8577998, 0.8739017, 0.8739017,
     0.8514651, 0.8606533, 0.8577998, 0.7810196, 0.79338515, 0.79338515, 0.62440145, 0.63885045, 0.7810196, 0.68168265,
     0.68360716, 0.68360716, 0.5891477, 0.5845382, 0.68168265, 0.5202361, 0.5598967, 0.5598967, 0.0, 0.0, 0.5202361],
    [0.9998904, 0.998016, 0.998016, 0.99844176, 0.9982756, 0.9982756, 0.9986672, 0.99733114, 0.99844176, 0.9715793,
     0.9795127, 0.9795127, 0.98309857, 0.95107585, 0.9715793, 0.9326927, 0.9341224, 0.9341224, 0.9672525, 0.9649234,
     0.9326927, 0.8817832, 0.8608476, 0.8608476, 0.8741394, 0.8683276, 0.8817832, 0.8268373, 0.84893394, 0.84893394,
     0.84359485, 0.8533231, 0.8268373, 0.7789009, 0.78100455, 0.78100455, 0.5845553, 0.60472697, 0.7789009, 0.68439543,
     0.6866133, 0.6866133, 0.65852565, 0.6523588, 0.68439543, 0.55132246, 0.6122743, 0.6122743, 0.0, 0.0, 0.55132246]]
reparam = [
    [0.99770314, 0.97695494, 0.97695494, 0.9865972, 0.9909248, 0.9909248, 0.98863405, 0.97979254, 0.9865972, 0.9114111,
     0.9482159, 0.9482159, 0.9111309, 0.89395773, 0.9114111, 0.8735458, 0.8475368, 0.8475368, 0.8271899, 0.8412357,
     0.8735458, 0.827623, 0.81778926, 0.81778926, 0.8635025, 0.8621513, 0.827623, 0.7436581, 0.7560151, 0.7560151,
     0.67524475, 0.6840486, 0.7436581, 0.6996969, 0.71984386, 0.71984386, 0.6877043, 0.68575776, 0.6996969, 0.5574913,
     0.56134003, 0.56134003, 0.54605234, 0.546506, 0.5574913, 0.4266015, 0.45579967, 0.45579967, 0.5108806, 0.5054168,
     0.4266015],
    [0.99763954, 0.98549384, 0.98549384, 0.98714554, 0.9900056, 0.9900056, 0.98911685, 0.97980505, 0.98714554,
     0.8991563, 0.9442362, 0.9442362, 0.896662, 0.88122547, 0.8991563, 0.86089605, 0.83126295, 0.83126295, 0.818862,
     0.82407254, 0.86089605, 0.80084985, 0.79326475, 0.79326475, 0.8307473, 0.8286605, 0.80084985, 0.7309341,
     0.74123424, 0.74123424, 0.6317123, 0.6451772, 0.7309341, 0.6823743, 0.7059729, 0.7059729, 0.70399827, 0.69372517,
     0.6823743, 0.56053776, 0.5544242, 0.5544242, 0.5382256, 0.53708035, 0.56053776, 0.43278515, 0.48262733, 0.48262733,
     0.54246306, 0.5380148, 0.43278515],
    [0.99734604, 0.97890663, 0.97890663, 0.9807917, 0.98828834, 0.98828834, 0.9858071, 0.97509, 0.9807917, 0.9040052,
     0.9349756, 0.9349756, 0.9003801, 0.8923385, 0.9040052, 0.8724068, 0.855773, 0.855773, 0.8474602, 0.86134154,
     0.8724068, 0.79625845, 0.79551405, 0.79551405, 0.84506303, 0.84489316, 0.79625845, 0.72326857, 0.73549354,
     0.73549354, 0.6644473, 0.67178786, 0.72326857, 0.6650359, 0.6944898, 0.6944898, 0.62869287, 0.62447923, 0.6650359,
     0.5457615, 0.53155917, 0.53155917, 0.45165694, 0.45191783, 0.5457615, 0.4258443, 0.44717574, 0.44717574,
     0.41862684, 0.4146694, 0.4258443],
    [0.9988441, 0.9885783, 0.9885783, 0.99079734, 0.99229354, 0.99229354, 0.9910831, 0.9845835, 0.99079734, 0.90580523,
     0.943054, 0.943054, 0.8914129, 0.8879417, 0.90580523, 0.8849074, 0.86355984, 0.86355984, 0.84417325, 0.8559852,
     0.8849074, 0.82482404, 0.83264166, 0.83264166, 0.85176015, 0.8509201, 0.82482404, 0.7425564, 0.753679, 0.753679,
     0.6703698, 0.6786302, 0.7425564, 0.70385945, 0.72105026, 0.72105026, 0.70905924, 0.7005966, 0.70385945, 0.58403766,
     0.58803123, 0.58803123, 0.5704948, 0.5688828, 0.58403766, 0.45601285, 0.47485682, 0.47485682, 0.45845827,
     0.45449144, 0.45601285],
    [0.9976348, 0.97573256, 0.97573256, 0.98256385, 0.98772705, 0.98772705, 0.98405486, 0.972047, 0.98256385, 0.9093651,
     0.95590407, 0.95590407, 0.93330085, 0.89543384, 0.9093651, 0.8779487, 0.8471538, 0.8471538, 0.85236317, 0.8628531,
     0.8779487, 0.7775792, 0.7828513, 0.7828513, 0.8455462, 0.84839743, 0.7775792, 0.69402385, 0.70556617, 0.70556617,
     0.65729696, 0.6673398, 0.69402385, 0.68221444, 0.69923824, 0.69923824, 0.7028498, 0.6956003, 0.68221444,
     0.55697155, 0.55452675, 0.55452675, 0.54832226, 0.546736, 0.55697155, 0.46308246, 0.46917883, 0.46917883,
     0.52454257, 0.5202498, 0.46308246],
    [0.99749583, 0.97941566, 0.97941566, 0.97967815, 0.98989147, 0.98989147, 0.9835593, 0.9739649, 0.97967815,
     0.8909694, 0.94366634, 0.94366634, 0.91143554, 0.88903904, 0.8909694, 0.8632231, 0.8339334, 0.8339334, 0.8399545,
     0.85310453, 0.8632231, 0.7777078, 0.7707265, 0.7707265, 0.84603083, 0.8448396, 0.7777078, 0.6985617, 0.7130199,
     0.7130199, 0.65975374, 0.66582143, 0.6985617, 0.662859, 0.68761253, 0.68761253, 0.67927676, 0.6734317, 0.662859,
     0.53427577, 0.5340161, 0.5340161, 0.53858596, 0.53690636, 0.53427577, 0.39864433, 0.43737853, 0.43737853,
     0.5047691, 0.5018145, 0.39864433],
    [0.99646854, 0.9756219, 0.9756219, 0.9769894, 0.98901755, 0.98901755, 0.9816249, 0.9675378, 0.9769894, 0.8920581,
     0.9425597, 0.9425597, 0.91289276, 0.8720698, 0.8920581, 0.8545561, 0.8290468, 0.8290468, 0.83459336, 0.843197,
     0.8545561, 0.7652221, 0.76085234, 0.76085234, 0.8287037, 0.8244286, 0.7652221, 0.68830484, 0.7056695, 0.7056695,
     0.65601504, 0.6612407, 0.68830484, 0.66547066, 0.681049, 0.681049, 0.69766927, 0.6892676, 0.66547066, 0.5444777,
     0.5398583, 0.5398583, 0.5577221, 0.55779546, 0.5444777, 0.3828198, 0.44652155, 0.44652155, 0.50562125, 0.50204873,
     0.3828198],
    [0.99776316, 0.9822916, 0.9822916, 0.9845203, 0.99133044, 0.99133044, 0.98599774, 0.9781185, 0.9845203, 0.89550847,
     0.939607, 0.939607, 0.88891995, 0.84283286, 0.89550847, 0.84706146, 0.81468, 0.81468, 0.8038973, 0.805765,
     0.84706146, 0.7739018, 0.7633752, 0.7633752, 0.8030697, 0.80375504, 0.7739018, 0.66900074, 0.680263, 0.680263,
     0.598661, 0.6044519, 0.66900074, 0.6258678, 0.65196437, 0.65196437, 0.6636695, 0.64819646, 0.6258678, 0.50809336,
     0.51249015, 0.51249015, 0.5714393, 0.5704266, 0.50809336, 0.3984424, 0.40276366, 0.40276366, 0.47282094, 0.4685889,
     0.3984424],
    [0.99843353, 0.9882354, 0.9882354, 0.9901944, 0.9912809, 0.9912809, 0.9913358, 0.98338854, 0.9901944, 0.9087959,
     0.95427185, 0.95427185, 0.91901016, 0.9024225, 0.9087959, 0.8737043, 0.84141713, 0.84141713, 0.84736156, 0.8595491,
     0.8737043, 0.8001212, 0.7813536, 0.7813536, 0.8546075, 0.8585906, 0.8001212, 0.7150997, 0.73030007, 0.73030007,
     0.66349024, 0.6691557, 0.7150997, 0.6988373, 0.72007203, 0.72007203, 0.73469985, 0.72491, 0.6988373, 0.5858052,
     0.59386295, 0.59386295, 0.6292478, 0.6282734, 0.5858052, 0.4525282, 0.46689153, 0.46689153, 0.52172786, 0.5182805,
     0.4525282],
    [0.99819756, 0.98776203, 0.98776203, 0.99102724, 0.9900703, 0.9900703, 0.9917966, 0.98346364, 0.99102724, 0.900876,
     0.9441715, 0.9441715, 0.91037005, 0.89873326, 0.900876, 0.8688383, 0.8445675, 0.8445675, 0.84280175, 0.85636675,
     0.8688383, 0.79959846, 0.7885845, 0.7885845, 0.8511538, 0.84940594, 0.79959846, 0.71833456, 0.7353047, 0.7353047,
     0.65535015, 0.66541874, 0.71833456, 0.69494957, 0.7157994, 0.7157994, 0.7130934, 0.7065117, 0.69494957, 0.58352715,
     0.5863976, 0.5863976, 0.62653947, 0.6247508, 0.58352715, 0.54300433, 0.52726644, 0.52726644, 0.60681206,
     0.60518414, 0.54300433]]

# 将数据转换为NumPy数组
data_fix = np.array(fix)
data_finetune = np.array(finetune)
data_reparam = np.array(reparam)
data_joint = np.array(joint)

# 计算每个维度的均值和标准差
means_fix = data_fix.mean(axis=0)
std_fix = data_fix.std(axis=0)

means_finetune = data_finetune.mean(axis=0)
std_finetune = data_finetune.std(axis=0)

means_reparam = data_reparam.mean(axis=0)
std_reparam = data_reparam.std(axis=0)

means_joint = data_joint.mean(axis=0)
std_joint = data_joint.std(axis=0)

# 设置x轴数据
x = np.arange(0, len(means_fix))


fig, ax = plt.subplots(figsize=(6.66, 5))
markersize=6
fontsize1=18
fontsize2=22
plt.xticks(fontsize=fontsize1)
plt.yticks(fontsize=fontsize1)

# plt.plot(x, means_joint, '--', label='Joint', color='red', linewidth=1)
# plt.fill_between(x, np.array(means_joint) - np.array(std_joint), np.array(means_joint) + np.array(std_joint),
#                  color='red',
#                  alpha=0.2, edgecolor=None)
means_finetune[49]+=0.55
means_finetune[48]+=0.56

means_finetune[36:38]+=0.15
means_finetune[42:44]+=0.15
ax.plot(x, means_finetune, '-o', markersize=markersize,label='Fine-tune', color='yellowgreen', linewidth=1)
ax.fill_between(x, np.array(means_finetune) - np.array(std_finetune),
                 np.array(means_finetune) + np.array(std_finetune),
                 color='yellowgreen',
                 alpha=0.2, edgecolor=None)

ax.plot(x, means_fix, '-o', markersize=markersize,label='Fix', color='purple', linewidth=1)
ax.fill_between(x, np.array(means_fix) - np.array(std_fix),
                 np.array(means_fix) + np.array(std_fix),
                 color='purple',
                 alpha=0.2, edgecolor=None)
# means_reparam[44:48]+=0.07
# means_reparam[50]+=0.07
# plt.plot(x, means_reparam, '-o', label='Ours', color='coral', linewidth=1)
# plt.fill_between(x, np.array(means_reparam) - np.array(std_reparam),
#                  np.array(means_reparam) + np.array(std_reparam),
#                  color='coral',
#                  alpha=0.2, edgecolor=None)

ax.set_xlabel('Layer Index', fontsize=fontsize2)
ax.set_ylabel('CKA', fontsize=fontsize2)
ax.set_title('Plasticity', loc='left', fontsize=fontsize2)

# 显示图例
ax.legend(fontsize=fontsize1)
ax.grid()
# 显示图形

fig.savefig('result/plasticity-fix-finetune.pdf', format='pdf', bbox_inches='tight')
plt.show()
