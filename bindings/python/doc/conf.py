import os
import re
import sys

try:
    import cntk
except ImportError:
    raise ImportError("Unable to import cntk; the cntk module needs to be built "
                      "and importable to generate documentation")

from cntk.sample_installer import module_is_unreleased

try:
    import sphinx_rtd_theme
except ImportError:
    raise ImportError("Unable to import sphinx_rtd_theme, please install via "
                      "'pip install sphinx_rtd_theme'")

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]

master_doc = 'index'

exclude_patterns = [
    '_build',
    'images',
    'test',
]

autodoc_mock_imports = [
    'tensorflow',
]

needs_sphinx = '1.5'

# TODO logging.

#suppress_warnings = [
#  'app.add_node',
#  'app.add_directive',
#  'app.add_role',
#  'app.add_generic_role',
#  'app.add_source_parser',
#  'image.data_uri',
#  'image.nonlocal_uri',
#  'ref.term',
#  'ref.ref',
#  'ref.numref',
#  'ref.keyword',
#  'ref.option',
#  'ref.citation',
#  'ref.doc',
#  'misc.highlighting_failure',
#  'toc.secnum',
#  'epub.unknown_project_files',
#]
## New in version 1.4.
## Changed in version 1.5: Added misc.highlighting_failure
## Changed in version 1.5.1: Added epub.unknown_project_files

nitpick_ignore = [
  ('py:obj', 'optional'),
  ('py:obj', ''), # TODO work around https://github.com/sphinx-doc/sphinx/issues/3320
  # TODO standardize on numpy.float{32,64}, make it linkable [but not exposed by numpy inventory]
  ('py:obj', 'np.float32'),
  ('py:obj', 'np.float64'),
  # TODO defaults should be move from the type hint to the description:
  ('py:obj', '0'),
  ('py:obj', '1'),
  ('py:obj', 'default'),
  ('py:obj', "default ''"),
  ('py:obj', "default to ''"),
  ('py:obj', "defaults to ''"),
  ('py:obj', "defaults to 0"),
  ('py:obj', "defaults to 1"),
  ('py:obj', 'default -1'),
  ('py:obj', 'default 0'),
  ('py:obj', 'default 0.0'),
  ('py:obj', 'default 0.00001'),
  ('py:obj', 'default 1'),
  ('py:obj', 'default 5000'),
  ('py:obj', 'default True'),
  ('py:obj', 'default False'),
  ('py:obj', "default 'center'"),
  ('py:obj', "default 'fill'"),
  ('py:obj', "default 'linear'"),
  ('py:obj', "default 'none'"),
  ('py:obj', "default 'placeholder'"),
  ('py:obj', 'default cntk.io.INFINITE_SAMPLES'),
  ('py:obj', 'default np.float32'),
  ('py:obj', 'default stdin'),
  ('py:obj', 'default stdout'),
  ('py:obj', 'default sys.exit'),
  ('py:obj', 'default None'),
  # TODO
  ('py:obj', 'NumPy Array'),
  ('py:obj', 'NumPy array'),
  ('py:obj', 'NumPy dtype'),
  ('py:obj', 'iterable'), # TODO should use :class:`~collections.abc.Iterable`
  ('py:obj', 'list of parameters'),
  ('py:obj', 'list of bools'),
  ('py:obj', 'list of lists of integers'),
  ('py:obj', 'Python function'),
  ('py:obj', 'a tuple of these'),
  ('py:obj', 'func'),
  ('py:obj', 'index'),
  ('py:obj', 'scalar'),
  ('py:obj', 'tensor without batch dimension;'),
  ('py:obj', 'value that can be cast to NumPy array'),
  ('py:obj', '0 arguments'),
  ('py:obj', 'BackPropState'),
  ('py:obj', 'Python function/lambda with 1'),
  ('py:obj', 'average_error'),
  ('py:obj', 'actual'),
  ('py:obj', 'graph node'),
  ('py:obj', 'keyword only'),
  ('py:obj', 'lambda'),
  ('py:obj', 'object behaving like sys.stdin'),
  ('py:obj', 'object behaving like sys.stdout'),
  ('py:obj', 'root node'),
  ('py:obj', 'a tuple thereof'),
  ('py:const', 'cntk.io.DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS'), # TODO write doc
  ('py:class', 'cntk.cntk_py.Dictionary'),
  ('py:obj', 'cntk.cntk_py.Dictionary'),
  ('py:class', 'cntk.cntk_py.StreamInformation'),
  ('py:class', 'cntk.cntk_py.minibatch_size_schedule'),
  ('py:class', 'cntk.cntk_py.ProgressWriter'),
  ('py:obj', 'cv_num_minibatches'),
  ('py:obj', 'cv_num_samples'),
  ('py:mod', 'cntk.utils'),
  ('py:class', 'cntk.Input'),
  ('py:class', 'cntk.variables.Variable.Type'),
  ('py:func', 'cntk.functions.Function.update_signature'),
  ('py:mod', 'cntk.utils'),
  ('py:obj', 'NumPy type'),
  ('py:class', 'cntk.input'),
  ('py:func', 'cntk.input_var'),
  ('py:obj', "float in case of 'dropoutRate'"),
  ('py:obj', "int for 'rngSeed'"),
  ('py:obj', 'function'),
  ('py:obj', 'progress writer'),
  ('py:obj', 'list of them'),
]

project = 'Python API for CNTK'
copyright = '2017, Microsoft'

version = cntk.__version__ # TODO consider shortening
release = cntk.__version__

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Do not prepend the current module to all description unit titles (such as ..
# function::).
add_module_names = False

# The theme to use for HTML and HTML Help pages.
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Linkcheck builder options
def re_exact_match(s):
  return re.compile('^' + re.escape(s) + '$')

linkcheck_anchors_ignore = [
  # Important: Github Wiki anchors (for sections refs) yield errors in
  # link-checking and need to be manually checked. Current exception to make
  # the build pass are listed here:
  re_exact_match('21-data-parallel-training-with-1-bit-sgd'),
  re_exact_match('22-block-momentum-sgd'),
  re_exact_match('converting-learning-rate-and-momentum-parameters-from-other-toolkits'),
  re_exact_match('for-python'),
  re_exact_match('base64imagedeserializer-options'),
]

source_prefix = 'https://github.com/Microsoft/CNTK/blob/'
if module_is_unreleased():
    source_prefix += 'master'
else:
    # TODO temporary
    source_prefix += 'v%s' % (cntk.__version__.replace("rc", ".rc"))

## TODO
def autodoc_process_docstring(app, what, name, obj, options, lines):
  pass
#  if name == 'cntk.axis.Axis':
#    import pdb; pdb.set_trace()

def autodoc_process_signature(app, what, name, obj, options, signature, return_annotation):
  if what == 'class': options['show-inheritance'] = False # TODO
  #http://www.sphinx-doc.org/en/stable/extdev/appapi.html#sphinx.application.Sphinx
  #app.warn("huch")
  #app.verbose
  #message, location=None, prefix='WARNING: ', type=None, subtype=None, colorfunc=<function inner>)
  #doctree-resolved(app, doctree, docname)
#  if name == 'cntk.axis.Axis':
#    import pdb; pdb.set_trace()

def autodoc_skip_member(app, what, name, obj, skip, options):
  return name == 'cntk_py' or skip
#  #import pdb; pdb.set_trace()
#  #return False
#
def setup(app):
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
    app.connect("autodoc-process-signature", autodoc_process_signature)
    app.connect("autodoc-skip-member", autodoc_skip_member)

# sphinx.ext.extlinks options
extlinks = {
    'cntk': (source_prefix + '/%s', ''),
    'cntktut': (source_prefix + '/Tutorials/%s.ipynb', ''),
    'cntkwiki': ('https://github.com/Microsoft/CNTK/wiki/%s', 'CNTK Wiki - ')
}

# sphinx.ext.intersphinx
# Note: to list an inventory's content: "python -m sphinx.ext.intersphinx objects.inv"
def intersphinx_target(base):
  # Use cached files if available
  if 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ:
     return os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'], 'Sphinx', base + '.inv')
  else:
    return None

intersphinx_mapping = {
  'python': ('https://docs.python.org/%s.%s' % sys.version_info[0:2],
             intersphinx_target('python-%s.%s' %  sys.version_info[0:2])),
  'numpy': ('https://docs.scipy.org/doc/numpy-1.11.0',
            intersphinx_target('numpy-1.11')),
  'scipy': ('https://docs.scipy.org/doc/scipy-0.17.1/reference',
            intersphinx_target('scipy-0.17.1')),
}

# sphinx.ext.napoleon options
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# sphinx.ext.todo options
todo_include_todos = module_is_unreleased()
