
# Cognitive Toolkit (CNTK) contrib

Code in this directory is only unofficially supported. The `contrib` directory contains project directories contributed by individual designated owner. The content in this directory would be merged into the core toolkit. During the merge process the contributed code will evolve with additional code for testing and documentation. Consequently, your original code can be changed or moved / removed at any time without notice.

One of the key goal is to minimize duplication within `contrib`. Consequently, some of your efforts to refactor the contributed code across multiple contributed directories or with the core library may be needed.

We encourage you to adhere to the following directory structure:
- Create a project directory in `contrib`: say `contrib/my_contrib/.`
- Mirror the portions of the CNTK code tree that your project requires

For example, if you were to  

For example,  you want to contribute two `GM_foo_example.py` and `GM_foo_example_test.py` for a novel generative model. If you were to merge those files directly into CNTK, they would live in `CNTK/Examples/Image/GM/GM_foo_example.py` and `CNTK/Tests/EndToEndTests/Image/GM/GM_foo_example_test.py`.  In `contrib` directory, they are part of project `GM`, and the full paths will be `contrib/GM/Examples/Image/GM/GM_foo_example.py` and `contrib/Tests/EndToEndTests/Image/GM/GM_foo_example_test.py`.
