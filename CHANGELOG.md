# Changelog

## [2.4.0](https://github.com/BlueBrain/TMD/compare/v2.3.1..2.4.0)

> 21 November 2023

### Build

- Drop support for Python 3.7 (Adrien Berchet - [#84](https://github.com/BlueBrain/TMD/pull/84))

### Chores And Housekeeping

- Remove some warnings (Adrien Berchet - [#86](https://github.com/BlueBrain/TMD/pull/86))

### General Changes

- Plot improvements (lidakanari - [#76](https://github.com/BlueBrain/TMD/pull/76))

## [v2.3.1](https://github.com/BlueBrain/TMD/compare/v2.3.0..v2.3.1)

> 16 May 2023

### New Features

- Files with upper-case extensions can be loaded (Adrien Berchet - [#78](https://github.com/BlueBrain/TMD/pull/78))

## [v2.3.0](https://github.com/BlueBrain/TMD/compare/v2.2.0..v2.3.0)

> 28 April 2023

### Chores And Housekeeping

- Update from Copier template (Adrien Berchet - [#73](https://github.com/BlueBrain/TMD/pull/73))
- Fix license detection (Adrien Berchet - [#59](https://github.com/BlueBrain/TMD/pull/59))
- Fix citation section and add CITATION.cff file (Adrien Berchet - [#58](https://github.com/BlueBrain/TMD/pull/58))
- Fix pylint config (Adrien Berchet - [#57](https://github.com/BlueBrain/TMD/pull/57))

### Documentation Changes

- Fix RTD (Adrien Berchet - [#56](https://github.com/BlueBrain/TMD/pull/56))

### Refactoring and Updates

- Make (x/y)lims and (x/y)lim consistent in plot functions (Adrien Berchet - [#69](https://github.com/BlueBrain/TMD/pull/69))
- Apply copier template (Adrien Berchet - [#55](https://github.com/BlueBrain/TMD/pull/55))

### Tidying of Code eg Whitespace

- Handle default kwargs (Adrien Berchet - [#67](https://github.com/BlueBrain/TMD/pull/67))

### CI Improvements

- Use Py37 with precommit (Adrien Berchet - [#72](https://github.com/BlueBrain/TMD/pull/72))
- Fix title in commitlint job (Adrien Berchet - [#70](https://github.com/BlueBrain/TMD/pull/70))
- Store tests results as artifacts (Adrien Berchet - [#66](https://github.com/BlueBrain/TMD/pull/66))
- Fix tests, bump Github actions and add get_by_name method to Population (Adrien Berchet - [#64](https://github.com/BlueBrain/TMD/pull/64))
- Fix action to publish new releases on Pypi (Adrien Berchet - [#53](https://github.com/BlueBrain/TMD/pull/53))

### General Changes

- Add new plotting functions to create overview figures for papers (lidakanari - [#63](https://github.com/BlueBrain/TMD/pull/63))
- Fix load_population to be able to load single files and to raise a proper exception (Adrien Berchet - [#54](https://github.com/BlueBrain/TMD/pull/54))

<!-- auto-changelog-above -->

## [v2.2.0](https://github.com/BlueBrain/TMD/compare/v2.1.0..v2.2.0)

> 22 August 2022

### General Changes

- Update to dendrite types (Alexis Arnaudon - [#52](https://github.com/BlueBrain/TMD/pull/52))
- Updated copyright year in several files (alex4200 - [#51](https://github.com/BlueBrain/TMD/pull/51))

## [v2.1.0](https://github.com/BlueBrain/TMD/compare/v2.0.11..v2.1.0)

> 9 March 2022

- Add trunk length feature (Adrien Berchet - [#44](https://github.com/BlueBrain/TMD/pull/44))
- Use points instead of diameters for MorphIO conversion (Adrien Berchet - [#47](https://github.com/BlueBrain/TMD/pull/47))
- Add missed coverage (Eleftherios Zisis - [#49](https://github.com/BlueBrain/TMD/pull/49))
- Refactor tests to use pytest & remove warnings (Eleftherios Zisis - [#48](https://github.com/BlueBrain/TMD/pull/48))
- Improve perf of find_apical_point_distance_smoothed (Adrien Berchet - [#46](https://github.com/BlueBrain/TMD/pull/46))
- Add tracking of points for bars (lidakanari - [#43](https://github.com/BlueBrain/TMD/pull/43))
- Fix license (Adrien Berchet - [#42](https://github.com/BlueBrain/TMD/pull/42))

## [v2.0.11](https://github.com/BlueBrain/TMD/compare/v2.0.10..v2.0.11)

> 18 March 2021

- bump morphio version (Alexis Arnaudon - [#41](https://github.com/BlueBrain/TMD/pull/41))

## [v2.0.10](https://github.com/BlueBrain/TMD/compare/v2.0.9..v2.0.10)

> 16 March 2021

- remove enum compat (Alexis Arnaudon - [#40](https://github.com/BlueBrain/TMD/pull/40))
- Move morphio loader to io module (lidakanari - [#38](https://github.com/BlueBrain/TMD/pull/38))
- Convert morphio Morphology to tmd Neuron (Eleftherios Zisis - [#37](https://github.com/BlueBrain/TMD/pull/37))
- Refactor Topology methods (Eleftherios Zisis - [#36](https://github.com/BlueBrain/TMD/pull/36))
- Create run-tox.yml (Alexis Arnaudon - [#34](https://github.com/BlueBrain/TMD/pull/34))
- Create publish-sdist.yml (Alexis Arnaudon - [#35](https://github.com/BlueBrain/TMD/pull/35))
- lint fixes + improvement of some older code (Alexis Arnaudon - [#32](https://github.com/BlueBrain/TMD/pull/32))

## [v2.0.9](https://github.com/BlueBrain/TMD/compare/v2.0.8..v2.0.9)

> 27 April 2020

- new_axes=False + soma outline fix (Alexis Arnaudon - [#30](https://github.com/BlueBrain/TMD/pull/30))

## [v2.0.8](https://github.com/BlueBrain/TMD/compare/v2.0.7..v2.0.8)

> 20 March 2020

- Add apical point function (lidakanari - [#29](https://github.com/BlueBrain/TMD/pull/29))

## [v2.0.7](https://github.com/BlueBrain/TMD/compare/v2.0.6..v2.0.7)

> 11 February 2020

- Update version to fix release (lidakanari - [#27](https://github.com/BlueBrain/TMD/pull/27))

- version = 2.0.6 (kanari - [8adba8d](https://github.com/BlueBrain/TMD/commit/8adba8d6d285fe8a4fed73203c723f3da84a7b63))

## [v2.0.6](https://github.com/BlueBrain/TMD/compare/v2.0.5..v2.0.6)

> 11 February 2020

- check if .asc files as input (Alexis Arnaudon - [#24](https://github.com/BlueBrain/TMD/pull/24))

- Update enum (kanari - [795a277](https://github.com/BlueBrain/TMD/commit/795a27703034d5376459b07c687e8de6193a3145))
- automate version update with releases (kanari - [ceb7437](https://github.com/BlueBrain/TMD/commit/ceb7437e8cebfb1643b7f641b79a05d9f57496f7))

## [v2.0.5](https://github.com/BlueBrain/TMD/compare/v2.0.4..v2.0.5)

> 17 January 2020

- Version increase  (lidakanari - [#23](https://github.com/BlueBrain/TMD/pull/23))
- Improve apical point definition (lidakanari - [#22](https://github.com/BlueBrain/TMD/pull/22))
- Move all tests in a single directory (Benoit Coste - [#19](https://github.com/BlueBrain/TMD/pull/19))
- Fix path distances consistency (lidakanari - [#15](https://github.com/BlueBrain/TMD/pull/15))
- Updated Tutorial.txt (lidakanari - [#12](https://github.com/BlueBrain/TMD/pull/12))
- Generalize population loader to accept list of files as input (lidakanari - [#6](https://github.com/BlueBrain/TMD/pull/6))

- Activate tox envs (Benoît Coste - [3774d12](https://github.com/BlueBrain/TMD/commit/3774d1258c4c818187c2ef04d11a1acb2495d75d))
- Fixes for review (kanari - [433470e](https://github.com/BlueBrain/TMD/commit/433470eef5ba823ab846ea6569325a52ce85e20e))
- Fix tests, activate tests for p27, pylint, pep8 (kanari - [7d4f427](https://github.com/BlueBrain/TMD/commit/7d4f4274696328e4cc3f02cb29c4aa253c33738d))
- This doesn't work (kanari - [010e53c](https://github.com/BlueBrain/TMD/commit/010e53ce685a0a3b7db11cf342ef409c31f6c006))
- Fix xrange for python 3 (kanari - [202354a](https://github.com/BlueBrain/TMD/commit/202354a9bddfc31777230ccfc38726b2508c2196))
- Update Tutorial.txt (lidakanari - [530ed57](https://github.com/BlueBrain/TMD/commit/530ed57fece137fe277e74e3dd33256616a78a32))
- Changed get_ph_diagram to get_persistence_diagram (Stanislav Schmidt - [8d8c952](https://github.com/BlueBrain/TMD/commit/8d8c952adba043f1595fa826ffca08efe8e8c50d))

## [v2.0.4](https://github.com/BlueBrain/TMD/compare/v2.0.3..v2.0.4)

> 3 March 2019

- Fix incopatibility with python3 (lidakanari - [#11](https://github.com/BlueBrain/TMD/pull/11))
- Deploy on each commit (Eleftherios Zisis - [#9](https://github.com/BlueBrain/TMD/pull/9))

- Implementation of closest barcode function (Eleftherios Zisis - [076ba9e](https://github.com/BlueBrain/TMD/commit/076ba9ee1d4242abcd33c42b46807f7b5c59977a))
- review corrections (Eleftherios Zisis - [ea9c93c](https://github.com/BlueBrain/TMD/commit/ea9c93ce58406f9f8321fc9cff33d13a1e045a98))
- Minor corrections (kanari - [23d9dd2](https://github.com/BlueBrain/TMD/commit/23d9dd2ca7b56254ec3f4275e9c2815930a95d39))
- lint corrections (Eleftherios Zisis - [3903168](https://github.com/BlueBrain/TMD/commit/3903168ad8e2c5f2d6573cf394a48505a0f6f919))
- Docstring clarification (Eleftherios Zisis - [c2955ff](https://github.com/BlueBrain/TMD/commit/c2955ffaf1fcd3065a4fee110b48df8dfb29ad80))
- removed commented code (Eleftherios Zisis - [bfc3494](https://github.com/BlueBrain/TMD/commit/bfc3494feb9b8a584068e300d3456e3959539f4a))
- import view should be import tmd.view as view (Oren Amsalem - [3f9a04d](https://github.com/BlueBrain/TMD/commit/3f9a04de07f11004ad60c207c01d372f13ea155b))
- Revert deploy on each commit (Benoît Coste - [c3e41f8](https://github.com/BlueBrain/TMD/commit/c3e41f85791dcb7c80852848e7d5440f0d4df33c))

## v2.0.3

> 16 January 2019

- Test fix Jenkins (Benoit Coste - [#2](https://github.com/BlueBrain/TMD/pull/2))
- Add Travis and Pypi integration (Benoit Coste - [#1](https://github.com/BlueBrain/TMD/pull/1))

- Cleaning up code (kanari - [7705675](https://github.com/BlueBrain/TMD/commit/77056758e713b9260e7e73f1aa2c8b28c75e01a0))
- Fix remaining pylint, pep8 issues (kanari - [8d809c3](https://github.com/BlueBrain/TMD/commit/8d809c3eeca92ca2db82e74df96c4ec288123a45))
- Further cleaning and removing unused functions (kanari - [5da070b](https://github.com/BlueBrain/TMD/commit/5da070b90d16d1d0c10ee4c66f036104670e23a9))
- Add basic functionality (kanari - [6f3af7c](https://github.com/BlueBrain/TMD/commit/6f3af7c6570455a223afe5bcc0a0e6665454a855))
- Minor modifications to basic functionality. Improve visualization. (kanari - [a5c52de](https://github.com/BlueBrain/TMD/commit/a5c52de975f258e7b7e0f88e545be13317abf779))
- Add appropriate license for open sourcing (Lida Kanari - [b9b3776](https://github.com/BlueBrain/TMD/commit/b9b37766fd499e5f33ff4ad4c1670ea2d682aacf))
- Fixing final pylinting issues; (kanari - [e5dfc6e](https://github.com/BlueBrain/TMD/commit/e5dfc6eabf0db22a54f79b1ca96c365a3a6b9e27))
- Correct pep8 (kanari - [56dca21](https://github.com/BlueBrain/TMD/commit/56dca2172db8602744f2e16cbbe4a523c98585f2))
- Optimize path_distance computation: use path_distances_2 (kanari - [b42c5de](https://github.com/BlueBrain/TMD/commit/b42c5de3cbd7404012ee5e8cfa40bbd93224ac46))
- Add pypi integration (Benoît Coste - [6ef1ff4](https://github.com/BlueBrain/TMD/commit/6ef1ff477f9fdbdd68e8d9fa2f02f3c48fb37852))
- Add apical point definition (kanari - [58b8b3c](https://github.com/BlueBrain/TMD/commit/58b8b3c12a996280e7d5d12ef8e03f22c587b041))
- Cleaning up and README update (Lida Kanari - [7975fe2](https://github.com/BlueBrain/TMD/commit/7975fe2682b5b54f4845eef80b8066716d880504))
- Readme corrections (Lida Kanari - [b702757](https://github.com/BlueBrain/TMD/commit/b702757708e4e6574813d2e9a2f69b487000a2e3))
- Update README.md (lidakanari - [8996afc](https://github.com/BlueBrain/TMD/commit/8996afc54ec6dfd61743f636ac5876969f72b1cd))
- Update README.md (lidakanari - [f3a01ea](https://github.com/BlueBrain/TMD/commit/f3a01ea0df423271c31069a967ca888f1a3a6bf3))
- Update README.md (lidakanari - [f4aee95](https://github.com/BlueBrain/TMD/commit/f4aee955e7abb723c8f483f7ec69b8e2cf00bcca))
- tmd/Topology/transformations.py (kanari - [bfccd1b](https://github.com/BlueBrain/TMD/commit/bfccd1b58de1802e1b8b0a27deefbbc9c3f6d24c))
- Make viewers optional (kanari - [8a87f84](https://github.com/BlueBrain/TMD/commit/8a87f84348909ac35cf868ccb922f61c37467f9c))
- Minor corrections (kanari - [2f0dc5b](https://github.com/BlueBrain/TMD/commit/2f0dc5b32a5e4fb3fb11114ebbe50758e5891155))
- More attempts on readme (Lida Kanari - [051af30](https://github.com/BlueBrain/TMD/commit/051af309168caad1c5522ec234da3091f671ac2b))
- Update Travis secret (Benoît Coste - [ffc1cff](https://github.com/BlueBrain/TMD/commit/ffc1cff100e5aa0e24346017ae32401a561b249c))
- Fix minor issue with apical distance definition" (kanari - [04f5dfa](https://github.com/BlueBrain/TMD/commit/04f5dfa747d72f16477a2f818d70d1c2bd98e1ea))
- missing _neuron in load command (nadavyayon - [50a1584](https://github.com/BlueBrain/TMD/commit/50a158420867d2dcbd7bb9d26810bc04cc8e212f))
- Update README.md (lidakanari - [b6e0ea8](https://github.com/BlueBrain/TMD/commit/b6e0ea8762026d45e38ce51bd75deb9840ee6346))
- Bump version (Benoît Coste - [7fb11db](https://github.com/BlueBrain/TMD/commit/7fb11db7848f12b51a7da3733f9132d0b28a77fb))
- fix docs (Benoît Coste - [159ad93](https://github.com/BlueBrain/TMD/commit/159ad93aad59cfb0eee170c0a6389664e1e2be68))
- Update README.md (lidakanari - [3bb4534](https://github.com/BlueBrain/TMD/commit/3bb45341d4bf88a7325242d02e3ad7a04f9bafa0))
- Update README.md (lidakanari - [8cc1bfa](https://github.com/BlueBrain/TMD/commit/8cc1bfa4b150cc2986de188b4f509d7a06f7dfa5))
- Rename LICENSE.txt to LICENSE.md (lidakanari - [22757f6](https://github.com/BlueBrain/TMD/commit/22757f68763fead8720189caaf1cc005748bb6b8))
- More attempts on readme (Lida Kanari - [bf1f36c](https://github.com/BlueBrain/TMD/commit/bf1f36c7e587a7eb736263f7c06d72df5f7364d0))
- Transfer Readme file (Lida Kanari - [1102051](https://github.com/BlueBrain/TMD/commit/110205119f49c49b2692f151c5b7389d2ac5f1be))
- Merge "Bump version" (Lida Kanari - [87b66a7](https://github.com/BlueBrain/TMD/commit/87b66a72481c205350abe8ac55445c481d86b20d))
- Merge "fix docs" (Lida Kanari - [11e00a2](https://github.com/BlueBrain/TMD/commit/11e00a255c095e7e4861da948ac67a021cf7a6f4))
- Migrating code from bitbucket (Lida Kanari - [01be170](https://github.com/BlueBrain/TMD/commit/01be170f89850620d12f452e36d738e6848e15b5))
- Add Tox file (Benoit Coste - [69333fb](https://github.com/BlueBrain/TMD/commit/69333fb1ccfb946c44b78cfac4ead52cde6bf943))
- Cleaning up code to simplify applications (kanari - [2f8a80b](https://github.com/BlueBrain/TMD/commit/2f8a80befcae464e50c657fef5b3e3a1db5a3ac9))
- Add examples and minor changes (kanari - [ab560ae](https://github.com/BlueBrain/TMD/commit/ab560aee9d77ef91860d85d755f01983834849b8))
- Clean up pep n pylint errors (Lida Kanari - [7e9f6b8](https://github.com/BlueBrain/TMD/commit/7e9f6b89ba8c67224083530c220ad47c61a40e12))
- Add matching with munkress (kanari - [2dbe26b](https://github.com/BlueBrain/TMD/commit/2dbe26b9d62fda42be9d2a04c901556015c8b5e8))
- Improvements and new plots (kanari - [f3e34ad](https://github.com/BlueBrain/TMD/commit/f3e34ad96d016571ca31109196485ddf7ec20405))
- Commit statistics as examples (kanari - [f9cca06](https://github.com/BlueBrain/TMD/commit/f9cca06561374ea250d7cc64f18e1d494b2c8b3c))
- Simplify matchings (kanari - [544e6fd](https://github.com/BlueBrain/TMD/commit/544e6fd06540d94a97757fb777347c3ccd983bb2))
- Add example to classify one cell against a population of cells (kanari - [e878a91](https://github.com/BlueBrain/TMD/commit/e878a9171c21366d1d61d27a144f10259abfd421))
- Average PI (kanari - [273b92b](https://github.com/BlueBrain/TMD/commit/273b92b22e9843c5d6995e5ce89aa530fa18e8d1))
- Add simple doc and remove unnecessary directories (Lida Kanari - [df48cc6](https://github.com/BlueBrain/TMD/commit/df48cc6c20cffd1b672132221c08efc04ad8acc7))
- Minor corrections (Lida Kanari - [823df3d](https://github.com/BlueBrain/TMD/commit/823df3d8f24938e8f7051fdb9558061c783e1e7f))
- Add polar plots code and example (kanari - [942fce0](https://github.com/BlueBrain/TMD/commit/942fce0a490c8a1b709f4c7d7370b9058a00b03b))
- Add version number in TMD installation (Benoit Coste - [c414aa2](https://github.com/BlueBrain/TMD/commit/c414aa202ad41eb8054def9d814dfcad237a3059))
- Update viewers (Lida Kanari - [d5ad47a](https://github.com/BlueBrain/TMD/commit/d5ad47a73033ac5c7beee002d85310cbc81cc791))
- Correct plotting functions (Lida Kanari - [5cf0b10](https://github.com/BlueBrain/TMD/commit/5cf0b1085b760f07385d1cd1a9f5a5a7acb08040))
- Update matching options (Lida Kanari - [d4d6cb3](https://github.com/BlueBrain/TMD/commit/d4d6cb390bc1ed0a862af2ee336a06df64248413))
- Add properties to Neuron, Population (Lida Kanari - [5380fd9](https://github.com/BlueBrain/TMD/commit/5380fd91402beb0ade5487039db6a31addbdabe8))
- Change the neurite_types inside a population to make consistent with neuron types (kanari - [57b7e70](https://github.com/BlueBrain/TMD/commit/57b7e7008c99cd4790655a4aaecf64384263da0e))
- Add examples and view (kanari - [8178c8e](https://github.com/BlueBrain/TMD/commit/8178c8e9b95f424917103f2cfbb6babd9f346056))
- Minor updates on viewers (Lida Kanari - [4230592](https://github.com/BlueBrain/TMD/commit/4230592a3882a6fc800463e0449cff26d5da1eee))
- Improve matching (kanari - [d9463f3](https://github.com/BlueBrain/TMD/commit/d9463f39093c5178222f102502c182132d04b4a1))
- Minor plotting improvements (Lida Kanari - [ef3d5c0](https://github.com/BlueBrain/TMD/commit/ef3d5c03ef3b221df360976e71441548ad313e3b))
- Replace pep8 (Lida Kanari - [8ea0ccc](https://github.com/BlueBrain/TMD/commit/8ea0ccc63a352e5522a1b67c50d23a2327577ec3))
- Allow optional definition of soma type (kanari - [f4cd02d](https://github.com/BlueBrain/TMD/commit/f4cd02d21ed8abd66e31b6d4f243180f98547f92))
- Minor simplification (Lida Kanari - [1252b86](https://github.com/BlueBrain/TMD/commit/1252b863eb79442426379179994a761549c8f0e6))
- Fix fail error message in population loader (kanari - [353f767](https://github.com/BlueBrain/TMD/commit/353f7672861a799ec43c941f6a9c4a1d4c9022c6))
- Update makefile (Lida Kanari - [dd502e3](https://github.com/BlueBrain/TMD/commit/dd502e3fae66b1d746c42d4f883af72801111e16))
- Fix error in previous commit (Lida Kanari - [ccb6db1](https://github.com/BlueBrain/TMD/commit/ccb6db179fe1154816f3cd9f42efb67c04237350))
- Data to check in h5 format (Lida Kanari - [a724985](https://github.com/BlueBrain/TMD/commit/a72498584875b5321c8cf4d7a951abc78efcc81a))
- Initial empty repository (Julien Francioli - [3ce483e](https://github.com/BlueBrain/TMD/commit/3ce483e09d1fe1e10f96fdba06e8303dd53cc4dd))
