FROM node:20-bookworm-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    sudo \
    ca-certificates \
    ripgrep \
    fd-find \
    jq \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

# Install beads (bd) — pinned to /usr/local/bin so it's on PATH for any user
RUN curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh \
        -o /tmp/install-bd.sh \
    && BEADS_INSTALL_DIR=/usr/local/bin bash /tmp/install-bd.sh \
    && rm /tmp/install-bd.sh \
    && bd --version

# Passwordless sudo for node + pre-create .claude with correct ownership
# so a freshly-mounted named volume inherits node:node instead of root:root
RUN echo "node ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p /home/node/.claude \
    && chown -R node:node /home/node/.claude

# Bake the Ralph loop into the image
COPY --chown=node:node ralph.sh /home/node/ralph.sh
RUN chmod +x /home/node/ralph.sh

USER node
WORKDIR /workspace

# tini handles signals cleanly so `cs stop` actually stops the process
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/home/node/ralph.sh"]
