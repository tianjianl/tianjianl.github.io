# frozen_string_literal: true

source "https://rubygems.org"
gemspec

# Ruby 3.0+ dropped webrick from the standard library; Jekyll 4.2's serve needs it.
gem "webrick", "~> 1.8"

# Pin Jekyll to the 4.2 line so it uses the sassc-based converter (the 4.4 line's
# sass-embedded native binary crashes with "Broken pipe" inside this container).
gem "jekyll", "~> 4.2.0"
gem "jekyll-sass-converter", "~> 2.0"
