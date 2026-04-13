// SpaceAce Map Editor
class MapEditor {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Map data
        this.vertices = [];
        this.lines = [];
        this.pickups = [];
        this.spawnVertex = null;
        this.mapWidth = 800;
        this.mapHeight = 600;
        
        // Editor state
        this.currentTool = 'vertex';
        this.selectedVertex = null;
        this.lineStart = null;
        this.isDragging = false;
        this.draggedVertex = null;
        this.isPanning = false;
        this.panStartX = 0;
        this.panStartY = 0;
        this.panStartOffsetX = 0;
        this.panStartOffsetY = 0;
        
        // View state
        this.zoom = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        
        // Initialize
        this.setupCanvas();
        this.setupEventListeners();
        this.loadLevelData();
        this.render();
    }
    
    setupCanvas() {
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }
    
    resizeCanvas() {
        const container = document.getElementById('canvas-container');
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
        this.render();
    }
    
    setupEventListeners() {
        // Tool buttons
        document.getElementById('tool-vertex').addEventListener('click', () => this.setTool('vertex'));
        document.getElementById('tool-line').addEventListener('click', () => this.setTool('line'));
        document.getElementById('tool-pickup').addEventListener('click', () => this.setTool('pickup'));
        document.getElementById('tool-spawn').addEventListener('click', () => this.setTool('spawn'));
        document.getElementById('tool-move').addEventListener('click', () => this.setTool('move'));
        document.getElementById('tool-delete').addEventListener('click', () => this.setTool('delete'));
        
        // Canvas events
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        this.canvas.addEventListener('wheel', (e) => this.onWheel(e));
        
        // Keyboard events
        document.addEventListener('keydown', (e) => this.onKeyDown(e));
        
        // Level select
        document.getElementById('level-select').addEventListener('change', (e) => {
            if (e.target.value === 'new') {
                document.getElementById('custom-level-number').style.display = 'block';
            } else {
                document.getElementById('custom-level-number').style.display = 'none';
                this.loadLevel(e.target.value);
            }
        });
        
        // Map resize
        document.getElementById('resize-map').addEventListener('click', () => this.resizeMap());
        
        // Actions
        document.getElementById('clear-map').addEventListener('click', () => this.clearMap());
        document.getElementById('export-btn').addEventListener('click', () => this.exportLevel());
        document.getElementById('import-btn').addEventListener('click', () => {
            document.getElementById('import-file').click();
        });
        document.getElementById('import-file').addEventListener('change', (e) => this.importLevel(e));
        
        // Zoom controls
        document.getElementById('zoom-in').addEventListener('click', () => this.zoomIn());
        document.getElementById('zoom-out').addEventListener('click', () => this.zoomOut());
        document.getElementById('zoom-fit').addEventListener('click', () => this.fitToView());
        document.getElementById('zoom-reset').addEventListener('click', () => this.resetView());
    }
    
    setTool(tool) {
        this.currentTool = tool;
        this.lineStart = null;
        this.selectedVertex = null;
        
        // Update button states
        document.querySelectorAll('#sidebar button').forEach(btn => {
            btn.classList.remove('active');
        });
        document.getElementById(`tool-${tool}`).classList.add('active');
        
        // Update cursor
        if (tool === 'move') {
            this.canvas.style.cursor = 'move';
        } else if (tool === 'delete') {
            this.canvas.style.cursor = 'not-allowed';
        } else {
            this.canvas.style.cursor = 'crosshair';
        }
        
        this.render();
    }
    
    worldToScreen(x, y) {
        return {
            x: (x - this.offsetX) * this.zoom + this.canvas.width / 2,
            y: (y - this.offsetY) * this.zoom + this.canvas.height / 2
        };
    }
    
    screenToWorld(x, y) {
        return {
            x: (x - this.canvas.width / 2) / this.zoom + this.offsetX,
            y: (y - this.canvas.height / 2) / this.zoom + this.offsetY
        };
    }
    
    onMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const screenX = e.clientX - rect.left;
        const screenY = e.clientY - rect.top;
        const worldPos = this.screenToWorld(screenX, screenY);
        
        // Middle mouse button for panning
        if (e.button === 1) {
            this.isPanning = true;
            this.panStartX = e.clientX;
            this.panStartY = e.clientY;
            this.panStartOffsetX = this.offsetX;
            this.panStartOffsetY = this.offsetY;
            e.preventDefault();
            return;
        }
        
        // Left mouse button for tools
        if (e.button === 0) {
            let actionTaken = false;
            
            switch (this.currentTool) {
                case 'vertex':
                    this.addVertex(worldPos.x, worldPos.y);
                    actionTaken = true;
                    break;
                    
                case 'line':
                    actionTaken = this.handleLineCreation(worldPos.x, worldPos.y);
                    break;
                    
                case 'pickup':
                    actionTaken = this.handlePickupPlacement(worldPos.x, worldPos.y);
                    break;
                    
                case 'spawn':
                    actionTaken = this.handleSpawnPlacement(worldPos.x, worldPos.y);
                    break;
                    
                case 'move':
                    // Try to grab a vertex first
                    const vertexIndex = this.findNearestVertex(worldPos.x, worldPos.y, 20 / this.zoom);
                    if (vertexIndex !== -1) {
                        this.startDragging(worldPos.x, worldPos.y);
                        actionTaken = true;
                    }
                    break;
                    
                case 'delete':
                    actionTaken = this.deleteElement(worldPos.x, worldPos.y);
                    break;
            }
            
            // If no action was taken, allow panning by dragging
            if (!actionTaken) {
                this.isPanning = true;
                this.panStartX = e.clientX;
                this.panStartY = e.clientY;
                this.panStartOffsetX = this.offsetX;
                this.panStartOffsetY = this.offsetY;
                this.canvas.style.cursor = 'grabbing';
            }
        }
    }
    
    onMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const screenX = e.clientX - rect.left;
        const screenY = e.clientY - rect.top;
        const worldPos = this.screenToWorld(screenX, screenY);
        
        // Update coordinate display
        document.getElementById('mouse-x').textContent = Math.round(worldPos.x);
        document.getElementById('mouse-y').textContent = Math.round(worldPos.y);
        
        // Handle panning
        if (this.isPanning) {
            const dx = (e.clientX - this.panStartX) / this.zoom;
            const dy = (e.clientY - this.panStartY) / this.zoom;
            this.offsetX = this.panStartOffsetX - dx;
            this.offsetY = this.panStartOffsetY - dy;
            this.render();
        }
        
        // Handle vertex dragging
        if (this.isDragging && this.draggedVertex !== null) {
            this.vertices[this.draggedVertex].x = worldPos.x;
            this.vertices[this.draggedVertex].y = worldPos.y;
            this.render();
        }
    }
    
    onMouseUp(e) {
        this.isDragging = false;
        this.draggedVertex = null;
        if (this.isPanning) {
            this.isPanning = false;
            // Reset cursor based on current tool
            if (this.currentTool === 'move') {
                this.canvas.style.cursor = 'move';
            } else if (this.currentTool === 'delete') {
                this.canvas.style.cursor = 'not-allowed';
            } else {
                this.canvas.style.cursor = 'crosshair';
            }
        }
    }
    
    onWheel(e) {
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        this.zoom *= delta;
        this.zoom = Math.max(0.1, Math.min(5, this.zoom));
        this.render();
    }
    
    onKeyDown(e) {
        const moveSpeed = 20 / this.zoom;
        let moved = false;
        
        switch(e.key) {
            case 'ArrowUp':
                this.offsetY -= moveSpeed;
                moved = true;
                break;
            case 'ArrowDown':
                this.offsetY += moveSpeed;
                moved = true;
                break;
            case 'ArrowLeft':
                this.offsetX -= moveSpeed;
                moved = true;
                break;
            case 'ArrowRight':
                this.offsetX += moveSpeed;
                moved = true;
                break;
        }
        
        if (moved) {
            e.preventDefault();
            this.render();
        }
    }
    
    addVertex(x, y) {
        this.vertices.push({ x, y });
        this.updateStats();
        this.render();
    }
    
    handleLineCreation(x, y) {
        const vertexIndex = this.findNearestVertex(x, y, 20 / this.zoom);
        
        if (vertexIndex !== -1) {
            if (this.lineStart === null) {
                this.lineStart = vertexIndex;
                this.selectedVertex = vertexIndex;
            } else {
                // Create line
                if (this.lineStart !== vertexIndex) {
                    this.lines.push({
                        start: this.lineStart,
                        end: vertexIndex
                    });
                    this.updateStats();
                }
                this.lineStart = null;
                this.selectedVertex = null;
            }
            this.render();
            return true;
        }
        return false;
    }
    
    handlePickupPlacement(x, y) {
        const vertexIndex = this.findNearestVertex(x, y, 20 / this.zoom);
        
        if (vertexIndex !== -1) {
            // Toggle pickup at this vertex
            const pickupIndex = this.pickups.indexOf(vertexIndex);
            if (pickupIndex === -1) {
                this.pickups.push(vertexIndex);
            } else {
                this.pickups.splice(pickupIndex, 1);
            }
            this.updateStats();
            this.render();
            return true;
        }
        return false;
    }
    
    handleSpawnPlacement(x, y) {
        const vertexIndex = this.findNearestVertex(x, y, 20 / this.zoom);
        console.log(`Spawn placement: Looking for vertex near (${x}, ${y}), found: ${vertexIndex}`);
        
        if (vertexIndex !== -1) {
            this.spawnVertex = vertexIndex;
            console.log(`Set spawn to vertex ${vertexIndex}`);
            this.updateStats();
            this.render();
            return true;
        }
        return false;
    }
    
    startDragging(x, y) {
        const vertexIndex = this.findNearestVertex(x, y, 20 / this.zoom);
        
        if (vertexIndex !== -1) {
            this.isDragging = true;
            this.draggedVertex = vertexIndex;
        }
    }
    
    deleteElement(x, y) {
        // Try to delete vertex
        const vertexIndex = this.findNearestVertex(x, y, 20 / this.zoom);
        
        if (vertexIndex !== -1) {
            // Remove vertex
            this.vertices.splice(vertexIndex, 1);
            
            // Update lines
            this.lines = this.lines.filter(line => 
                line.start !== vertexIndex && line.end !== vertexIndex
            );
            
            // Update indices in lines
            this.lines.forEach(line => {
                if (line.start > vertexIndex) line.start--;
                if (line.end > vertexIndex) line.end--;
            });
            
            // Update pickups
            this.pickups = this.pickups.filter(p => p !== vertexIndex);
            this.pickups = this.pickups.map(p => p > vertexIndex ? p - 1 : p);
            
            // Update spawn
            if (this.spawnVertex === vertexIndex) {
                this.spawnVertex = null;
            } else if (this.spawnVertex > vertexIndex) {
                this.spawnVertex--;
            }
            
            this.updateStats();
            this.render();
            return true;
        }
        
        // Try to delete line
        const lineIndex = this.findNearestLine(x, y, 10 / this.zoom);
        if (lineIndex !== -1) {
            this.lines.splice(lineIndex, 1);
            this.updateStats();
            this.render();
            return true;
        }
        
        return false;
    }
    
    findNearestVertex(x, y, threshold) {
        let minDist = threshold;
        let nearestIndex = -1;
        
        this.vertices.forEach((vertex, index) => {
            const dist = Math.sqrt(
                Math.pow(vertex.x - x, 2) + 
                Math.pow(vertex.y - y, 2)
            );
            
            if (dist < minDist) {
                minDist = dist;
                nearestIndex = index;
            }
        });
        
        return nearestIndex;
    }
    
    findNearestLine(x, y, threshold) {
        let minDist = threshold;
        let nearestIndex = -1;
        
        this.lines.forEach((line, index) => {
            const start = this.vertices[line.start];
            const end = this.vertices[line.end];
            
            if (start && end) {
                const dist = this.pointToLineDistance(x, y, start.x, start.y, end.x, end.y);
                
                if (dist < minDist) {
                    minDist = dist;
                    nearestIndex = index;
                }
            }
        });
        
        return nearestIndex;
    }
    
    pointToLineDistance(px, py, x1, y1, x2, y2) {
        const A = px - x1;
        const B = py - y1;
        const C = x2 - x1;
        const D = y2 - y1;
        
        const dot = A * C + B * D;
        const lenSq = C * C + D * D;
        let param = -1;
        
        if (lenSq !== 0) {
            param = dot / lenSq;
        }
        
        let xx, yy;
        
        if (param < 0) {
            xx = x1;
            yy = y1;
        } else if (param > 1) {
            xx = x2;
            yy = y2;
        } else {
            xx = x1 + param * C;
            yy = y1 + param * D;
        }
        
        const dx = px - xx;
        const dy = py - yy;
        
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    resizeMap() {
        this.mapWidth = parseInt(document.getElementById('map-width').value);
        this.mapHeight = parseInt(document.getElementById('map-height').value);
        this.render();
    }
    
    clearMap() {
        if (confirm('Are you sure you want to clear the map?')) {
            this.vertices = [];
            this.lines = [];
            this.pickups = [];
            this.spawnVertex = null;
            this.updateStats();
            this.render();
        }
    }
    
    zoomIn() {
        this.zoom *= 1.2;
        this.zoom = Math.min(5, this.zoom);
        this.render();
    }
    
    zoomOut() {
        this.zoom *= 0.8;
        this.zoom = Math.max(0.1, this.zoom);
        this.render();
    }
    
    resetView() {
        this.zoom = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.render();
    }
    
    fitToView() {
        if (this.vertices.length === 0) {
            this.resetView();
            return;
        }
        
        // Find bounds of all vertices
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        
        this.vertices.forEach(vertex => {
            minX = Math.min(minX, vertex.x);
            maxX = Math.max(maxX, vertex.x);
            minY = Math.min(minY, vertex.y);
            maxY = Math.max(maxY, vertex.y);
        });
        
        // Add padding
        const padding = 100;
        minX -= padding;
        maxX += padding;
        minY -= padding;
        maxY += padding;
        
        // Calculate zoom to fit
        const levelWidth = maxX - minX;
        const levelHeight = maxY - minY;
        
        const zoomX = this.canvas.width / levelWidth;
        const zoomY = this.canvas.height / levelHeight;
        
        this.zoom = Math.min(zoomX, zoomY) * 0.9; // 90% to leave some margin
        this.zoom = Math.max(0.1, Math.min(5, this.zoom));
        
        // Center the view
        this.offsetX = (minX + maxX) / 2;
        this.offsetY = (minY + maxY) / 2;
        
        this.render();
    }
    
    render() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Save context
        this.ctx.save();
        
        // Draw map bounds
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = 2;
        const topLeft = this.worldToScreen(0, 0);
        const bottomRight = this.worldToScreen(this.mapWidth, this.mapHeight);
        this.ctx.strokeRect(
            topLeft.x, topLeft.y,
            bottomRight.x - topLeft.x,
            bottomRight.y - topLeft.y
        );
        
        // Draw lines
        this.ctx.strokeStyle = '#fff';
        this.ctx.lineWidth = 2;
        this.lines.forEach(line => {
            const start = this.vertices[line.start];
            const end = this.vertices[line.end];
            
            if (start && end) {
                const screenStart = this.worldToScreen(start.x, start.y);
                const screenEnd = this.worldToScreen(end.x, end.y);
                
                this.ctx.beginPath();
                this.ctx.moveTo(screenStart.x, screenStart.y);
                this.ctx.lineTo(screenEnd.x, screenEnd.y);
                this.ctx.stroke();
            }
        });
        
        // Draw vertices
        this.vertices.forEach((vertex, index) => {
            const screen = this.worldToScreen(vertex.x, vertex.y);
            
            // Determine color and size
            let radius = 5;
            if (index === this.spawnVertex) {
                this.ctx.fillStyle = '#00ff00'; // Green for spawn
                radius = 8; // Larger for spawn
                
                // Draw outer ring for spawn
                this.ctx.strokeStyle = '#00ff00';
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.arc(screen.x, screen.y, radius + 3, 0, Math.PI * 2);
                this.ctx.stroke();
            } else if (this.pickups.includes(index)) {
                this.ctx.fillStyle = '#ffff00'; // Yellow for pickups
            } else if (index === this.selectedVertex) {
                this.ctx.fillStyle = '#ff00ff'; // Magenta for selected
            } else {
                this.ctx.fillStyle = '#ff0000'; // Red for normal vertices
            }
            
            // Draw vertex
            this.ctx.beginPath();
            this.ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Draw vertex index
            this.ctx.fillStyle = '#fff';
            this.ctx.font = '10px Arial';
            this.ctx.fillText(index.toString(), screen.x + 8, screen.y - 8);
            
            // Draw spawn label
            if (index === this.spawnVertex) {
                this.ctx.fillStyle = '#00ff00';
                this.ctx.font = 'bold 12px Arial';
                this.ctx.fillText('SPAWN', screen.x - 20, screen.y + 25);
            }
        });
        
        // Draw spawn indicator
        if (this.spawnVertex !== null && this.vertices[this.spawnVertex]) {
            const spawnPos = this.vertices[this.spawnVertex];
            const screen = this.worldToScreen(spawnPos.x, spawnPos.y - 100);
            
            this.ctx.strokeStyle = '#00ff00';
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.moveTo(screen.x - 10, screen.y);
            this.ctx.lineTo(screen.x + 10, screen.y);
            this.ctx.moveTo(screen.x, screen.y - 10);
            this.ctx.lineTo(screen.x, screen.y + 10);
            this.ctx.stroke();
            
            this.ctx.fillStyle = '#00ff00';
            this.ctx.font = '12px Arial';
            this.ctx.fillText('SPAWN', screen.x + 15, screen.y);
        }
        
        // Restore context
        this.ctx.restore();
    }
    
    
    updateStats() {
        document.getElementById('vertex-count').textContent = this.vertices.length;
        document.getElementById('line-count').textContent = this.lines.length;
        document.getElementById('pickup-count').textContent = this.pickups.length;
        document.getElementById('spawn-info').textContent = 
            this.spawnVertex !== null ? `Vertex ${this.spawnVertex}` : 'Not set';
    }
    
    exportLevel() {
        let levelId = document.getElementById('level-select').value;
        
        if (levelId === 'new') {
            const customLevel = document.getElementById('custom-level-number').value;
            if (!customLevel) {
                alert('Please enter a level number for the new level.');
                return;
            }
            levelId = customLevel;
        }
        
        // Load existing levels first
        fetch('spaceace_levels.json')
            .then(response => response.json())
            .then(levels => {
                // Update the current level with new data
                levels[levelId] = this.generateLevelData();
                
                // Format the complete levels object
                const json = JSON.stringify(levels, null, 2);
                
                // Display in export area
                document.getElementById('export-json').textContent = `Updated level ${levelId}. Copy this JSON and save it to spaceace_levels.json:\n\n${json}`;
                
                // Download as file
                const blob = new Blob([json], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'spaceace_levels.json';
                a.click();
                URL.revokeObjectURL(url);
                
                console.log(`Exported level ${levelId} to spaceace_levels.json`);
            })
            .catch(err => {
                // If can't load existing levels, create new structure
                const levels = {};
                levels[levelId] = this.generateLevelData();
                
                const json = JSON.stringify(levels, null, 2);
                
                // Display in export area
                document.getElementById('export-json').textContent = `Created new level ${levelId}. Copy this JSON and save it to spaceace_levels.json:\n\n${json}`;
                
                // Download as file
                const blob = new Blob([json], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'spaceace_levels.json';
                a.click();
                URL.revokeObjectURL(url);
                
                console.log(`Created new spaceace_levels.json with level ${levelId}`);
            });
    }
    
    generateLevelData() {
        const data = [];
        
        // Vertex count
        data.push(this.vertices.length);
        
        // Vertices
        this.vertices.forEach(vertex => {
            data.push(vertex.x, vertex.y);
        });
        
        // Line count
        data.push(this.lines.length);
        
        // Lines
        this.lines.forEach(line => {
            data.push(line.start, line.end);
        });
        
        // Start index
        data.push(this.spawnVertex !== null ? this.spawnVertex : 0);
        
        // Padding value (appears to be unused but required by format)
        data.push(0);
        
        // Map dimensions
        data.push(this.mapWidth, this.mapHeight);
        
        // Pickup count
        data.push(this.pickups.length);
        
        // Pickups
        this.pickups.forEach(pickup => {
            data.push(pickup);
        });
        
        // Triangle count (not used in editor, but required by format)
        data.push(0);
        
        return data;
    }
    
    importLevel(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const data = JSON.parse(event.target.result);
                this.parseLevelData(data);
            } catch (err) {
                alert('Invalid level file format');
            }
        };
        reader.readAsText(file);
    }
    
    loadLevel(levelId) {
        if (levelId === 'new') {
            this.clearMap();
            return;
        }
        
        // Load from existing levels
        fetch('spaceace_levels.json')
            .then(response => response.json())
            .then(levels => {
                if (levels[levelId]) {
                    this.parseLevelData(levels[levelId]);
                }
            })
            .catch(err => {
                console.error('Failed to load level data:', err);
            });
    }
    
    parseLevelData(data) {
        let index = 0;
        
        // Clear existing data
        this.vertices = [];
        this.lines = [];
        this.pickups = [];
        this.spawnVertex = null;
        
        // Parse vertices
        const vertexCount = data[index++];
        console.log(`Loading level with ${vertexCount} vertices`);
        
        for (let i = 0; i < vertexCount; i++) {
            this.vertices.push({
                x: data[index++],
                y: data[index++]
            });
        }
        
        // Parse lines
        const lineCount = data[index++];
        console.log(`Loading ${lineCount} lines`);
        
        for (let i = 0; i < lineCount; i++) {
            this.lines.push({
                start: data[index++],
                end: data[index++]
            });
        }
        
        // Parse spawn
        this.spawnVertex = data[index++];
        console.log(`Spawn vertex: ${this.spawnVertex}`);
        
        // Skip padding value
        if (index < data.length) {
            index++; // Skip padding
        }
        
        // Parse map dimensions
        if (index < data.length - 1) {
            this.mapWidth = data[index++];
            this.mapHeight = data[index++];
            document.getElementById('map-width').value = this.mapWidth;
            document.getElementById('map-height').value = this.mapHeight;
            console.log(`Map dimensions: ${this.mapWidth}x${this.mapHeight}`);
        }
        
        // Parse pickups
        if (index < data.length) {
            const pickupCount = data[index++];
            console.log(`Loading ${pickupCount} pickups`);
            
            for (let i = 0; i < pickupCount; i++) {
                if (index < data.length) {
                    this.pickups.push(data[index++]);
                }
            }
        }
        
        // Auto-fit view to show entire level
        this.fitToView();
        
        this.updateStats();
        this.render();
    }
    
    loadLevelData() {
        // Try to load level data for dropdown
        fetch('spaceace_levels.json')
            .then(response => response.json())
            .then(levels => {
                console.log('Levels loaded successfully');
            })
            .catch(err => {
                console.log('Could not load level data:', err);
            });
    }
}

// Initialize editor when page loads
window.addEventListener('load', () => {
    const editor = new MapEditor();
});