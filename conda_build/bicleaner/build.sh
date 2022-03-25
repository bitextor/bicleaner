
export PIP_NO_INDEX="False" # We are downloading requisites from PyPi
export PIP_NO_DEPENDENCIES="False" # We need the dependencies from our defined dependencies
export PIP_IGNORE_INSTALLED="False" # We need to take into account the dependencies
export PIP_NO_BUILD_ISOLATION="True"

if [[ ! -f $PREFIX/lib/libhunspell.so ]]; then
  ln -s $PREFIX/lib/libhunspell{-1.7,}.so
fi
if [[ ! -f $PREFIX/lib/libhunspell.a ]]; then
  ln -s $PREFIX/lib/libhunspell{-1.7,}.a
fi

# FastSpell (bicleaner-hardrules)
INCLUDE_PATH="$PREFIX/include" pip3 install hunspell

$PYTHON -m pip install .