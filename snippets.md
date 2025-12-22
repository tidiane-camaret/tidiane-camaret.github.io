# Install Ruby and Bundler
sudo apt update
sudo apt install ruby-full build-essential zlib1g-dev

# Avoid installing gems as root - configure gem installation path
echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Install Bundler
gem install bundler

# Install Jekyll (if needed)
gem install jekyll

# Serve with live reload (watches for changes)
bundle exec jekyll serve --livereload

# Serve on a different port
bundle exec jekyll serve --port 4001

# Make accessible from other devices on network
bundle exec jekyll serve --host 0.0.0.0

# Build drafts
bundle exec jekyll serve --drafts

# Incremental build (faster rebuilds)
bundle exec jekyll serve --incremental
