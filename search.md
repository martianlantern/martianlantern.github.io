---
layout: default
title: Search
---
<div class="col-sm-12">
    <div class="search-page">
        <div class="search-container">
            <input type="search" class="search-input" placeholder="Search posts..." id="searchbox" autofocus>
        </div>
        <div class="search-results" id="search-results"></div>
    </div>
</div>

<script>
(function() {
    var searchData = [];
    var searchInput = document.getElementById('searchbox');
    var searchResults = document.getElementById('search-results');

    // Load search data
    fetch('{{ site.url }}/search.json')
        .then(response => response.json())
        .then(data => {
            searchData = data;
        })
        .catch(err => console.error('Error loading search data:', err));

    // Debounce function
    function debounce(func, wait) {
        var timeout;
        return function() {
            var context = this, args = arguments;
            clearTimeout(timeout);
            timeout = setTimeout(function() {
                func.apply(context, args);
            }, wait);
        };
    }

    // Find all match positions in content
    function findAllMatches(content, query) {
        var matches = [];
        var lowerContent = content.toLowerCase();
        var lowerQuery = query.toLowerCase();
        var index = 0;
        
        while ((index = lowerContent.indexOf(lowerQuery, index)) !== -1) {
            matches.push(index);
            index += lowerQuery.length;
        }
        
        return matches;
    }

    // Get snippet around a specific position
    function getSnippetAt(content, position, query, snippetLength) {
        snippetLength = snippetLength || 120;
        
        var start = Math.max(0, position - 50);
        var end = Math.min(content.length, position + query.length + 70);
        
        // Try to start at word boundary
        if (start > 0) {
            var spaceIndex = content.lastIndexOf(' ', start + 10);
            if (spaceIndex > start - 20) start = spaceIndex + 1;
        }
        
        // Try to end at word boundary
        if (end < content.length) {
            var spaceIndex = content.indexOf(' ', end - 10);
            if (spaceIndex !== -1 && spaceIndex < end + 20) end = spaceIndex;
        }
        
        var snippet = '';
        if (start > 0) snippet += '...';
        snippet += content.substring(start, end).trim();
        if (end < content.length) snippet += '...';
        
        // Highlight the match
        var regex = new RegExp('(' + escapeRegex(query) + ')', 'gi');
        snippet = snippet.replace(regex, '<mark>$1</mark>');
        
        return snippet;
    }

    // Escape special regex characters
    function escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    // Perform search
    function performSearch(query) {
        if (!query || query.length < 2) {
            searchResults.innerHTML = '';
            return;
        }

        var results = [];
        var lowerQuery = query.toLowerCase();
        var totalMatches = 0;

        searchData.forEach(function(post) {
            var titleMatches = findAllMatches(post.title, query);
            var contentMatches = findAllMatches(post.content, query);
            
            if (titleMatches.length > 0 || contentMatches.length > 0) {
                var snippets = [];
                
                // Get unique snippets (avoid overlapping)
                var usedPositions = [];
                contentMatches.forEach(function(pos) {
                    // Check if this position is far enough from already used ones
                    var isFarEnough = usedPositions.every(function(usedPos) {
                        return Math.abs(pos - usedPos) > 100;
                    });
                    
                    if (isFarEnough && snippets.length < 5) { // Max 5 snippets per post
                        snippets.push(getSnippetAt(post.content, pos, query));
                        usedPositions.push(pos);
                    }
                });
                
                totalMatches += titleMatches.length + contentMatches.length;
                
                results.push({
                    title: post.title,
                    url: post.url,
                    date: post.date,
                    titleMatch: titleMatches.length > 0,
                    matchCount: titleMatches.length + contentMatches.length,
                    snippets: snippets
                });
            }
        });

        // Sort: most matches first, then title matches
        results.sort(function(a, b) {
            if (a.titleMatch && !b.titleMatch) return -1;
            if (!a.titleMatch && b.titleMatch) return 1;
            return b.matchCount - a.matchCount;
        });

        displayResults(results, query, totalMatches);
    }

    // Display results
    function displayResults(results, query, totalMatches) {
        if (results.length === 0) {
            searchResults.innerHTML = '<div class="no-results">No results found for "' + escapeHtml(query) + '"</div>';
            return;
        }

        var html = '<div class="results-count">' + totalMatches + ' match' + (totalMatches === 1 ? '' : 'es') + ' in ' + results.length + ' post' + (results.length === 1 ? '' : 's') + '</div>';
        html += '<div class="results-list">';
        
        results.forEach(function(result) {
            var highlightedTitle = result.title.replace(
                new RegExp('(' + escapeRegex(query) + ')', 'gi'),
                '<mark>$1</mark>'
            );
            
            html += '<div class="result-item">';
            html += '<a href="' + result.url + '" class="result-header">';
            html += '<span class="result-title">' + highlightedTitle + '</span>';
            html += '<span class="result-meta">' + result.date + ' · ' + result.matchCount + ' match' + (result.matchCount === 1 ? '' : 'es') + '</span>';
            html += '</a>';
            
            if (result.snippets.length > 0) {
                html += '<div class="result-snippets">';
                result.snippets.forEach(function(snippet) {
                    html += '<a href="' + result.url + '" class="result-snippet">' + snippet + '</a>';
                });
                html += '</div>';
            }
            
            html += '</div>';
        });
        
        html += '</div>';
        searchResults.innerHTML = html;
    }

    // Escape HTML
    function escapeHtml(text) {
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Add event listener with debounce
    searchInput.addEventListener('input', debounce(function(e) {
        performSearch(e.target.value.trim());
    }, 200));

    // Handle Enter key
    searchInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            var firstResult = searchResults.querySelector('.result-header');
            if (firstResult) {
                window.location.href = firstResult.href;
            }
        }
    });
})();
</script>
