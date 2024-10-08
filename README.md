# My Project
In this repo, I'll be experimenting with dimension reduction techniques on the training data used in my project Image-based classification of intense radio bursts from spectrograms: An application to Saturn Kilometric Radiation' published to the [Journal of Geophysical Research](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023JA031926), the repo is found [here](https://github.com/elodwyer1/Unet_Application_to_Saturn_Kilometric_Radiation/tree/main).

I'll be applying different dimension reduction techniques on a set of 1533 frequency-time arrays and comparing how their distributions differentiates between the sub-classes of radio burst. These subclasses are described in detail in this [paper](https://www.tara.tcd.ie/handle/2262/103103). The radio bursts we refer to are 'Low Frequency Extensions of Saturn Kilometric Radiation' (LFE). The subclasses of LFE are listed below.

| Type   | Count | Description                                                                 |
|--------|-------|-----------------------------------------------------------------------------|
| LFE    | 479   | Standard appearance                                                         |
| LFEm   | 96    | LFEs longer than a single planetary rotation                                 |
| LFEsp  | 99    | LFE with sparse emission                                                    |
| LFEsm  | 164   | LFE that is of intermediate extension, excursion to > 40kHz but < 100kHz     |
| LFEdg  | 111   | LFE with a datagap                                                          |
| LFEext | 35    | LFE with extinction at high frequencies                                     |

