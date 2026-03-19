const readmePath = '../README.md';

const categoryFilter = document.getElementById('categoryFilter');
const sourceFilter = document.getElementById('sourceFilter');
const searchInput = document.getElementById('searchInput');
const summary = document.getElementById('summary');
const worksGrid = document.getElementById('worksGrid');
const template = document.getElementById('workCardTemplate');

const sourceLabelMap = {
  paper: 'Paper',
  code: 'Code',
  blog: 'Blog'
};

async function boot() {
  try {
    const res = await fetch(readmePath);
    const markdown = await res.text();
    const works = parseWorks(markdown);
    hydrateCategoryFilter(works);
    bindEvents(works);
    render(works);
  } catch (err) {
    worksGrid.innerHTML = `<div class="empty">加载失败：${err.message}</div>`;
  }
}

function bindEvents(works) {
  [categoryFilter, sourceFilter, searchInput].forEach((el) => {
    el.addEventListener('input', () => render(works));
  });
}

function hydrateCategoryFilter(works) {
  const categories = [...new Set(works.map((item) => item.category))];
  for (const category of categories) {
    const option = document.createElement('option');
    option.value = category;
    option.textContent = category;
    categoryFilter.appendChild(option);
  }
}

function render(works) {
  const query = searchInput.value.trim().toLowerCase();
  const category = categoryFilter.value;
  const sourceType = sourceFilter.value;

  const filtered = works.filter((item) => {
    const hitCategory = category === 'all' || item.category === category;
    const hitSource =
      sourceType === 'all' || item.sources.some((source) => source.type === sourceType);
    const blob = [
      item.title,
      item.authors,
      item.institution,
      item.task,
      item.category
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase();
    const hitQuery = !query || blob.includes(query);
    return hitCategory && hitSource && hitQuery;
  });

  summary.textContent = `共 ${works.length} 项工作，当前显示 ${filtered.length} 项。`;

  if (!filtered.length) {
    worksGrid.innerHTML = '<div class="empty">没有匹配结果，请调整筛选条件。</div>';
    return;
  }

  worksGrid.innerHTML = '';
  for (const work of filtered) {
    const node = template.content.cloneNode(true);
    node.querySelector('.card__category').textContent = work.category;
    node.querySelector('.card__title').textContent = work.title;
    node.querySelector('.card__meta').innerHTML = [
      work.authors ? `<p><strong>Authors:</strong> ${work.authors}</p>` : '',
      work.institution ? `<p><strong>Institution:</strong> ${work.institution}</p>` : '',
      work.task ? `<p><strong>Task:</strong> ${work.task}</p>` : ''
    ].join('');

    const sourceWrap = node.querySelector('.card__sources');
    work.sources.forEach((source) => {
      const link = document.createElement('a');
      link.className = `badge badge--${source.type}`;
      link.href = source.url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = sourceLabelMap[source.type] || source.type;
      sourceWrap.appendChild(link);
    });

    worksGrid.appendChild(node);
  }
}

function parseWorks(markdown) {
  const lines = markdown.split('\n');
  const works = [];

  let currentCategory = '';
  let pending = null;

  for (const rawLine of lines) {
    const line = rawLine.trim();

    if (line.startsWith('### ') || line.startsWith('## ')) {
      const heading = line.replace(/^#+\s*/, '').trim();
      if (!['📖 Benchmarks', '🔧 Method'].includes(heading)) {
        currentCategory = heading;
      }
      continue;
    }

    if (line.startsWith('- **')) {
      const titleMatch = line.match(/-\s*\*\*(.*?)\*\*/);
      const title = titleMatch ? titleMatch[1].trim() : 'Untitled';
      const sources = extractSources(line);
      pending = {
        title,
        category: currentCategory || 'Uncategorized',
        authors: '',
        institution: '',
        task: '',
        sources
      };
      works.push(pending);
      continue;
    }

    if (!pending) continue;

    if (line.startsWith('- **Institution:**')) {
      pending.institution = line.replace('- **Institution:**', '').trim();
      continue;
    }

    if (line.startsWith('- **Task:**')) {
      pending.task = line.replace('- **Task:**', '').trim();
      continue;
    }

    if (line.startsWith('-') && !line.startsWith('- **Institution:**') && !line.startsWith('- **Task:**')) {
      pending.authors = line.replace(/^-\s*/, '').trim();
    }
  }

  return works.filter((work) => work.title && work.title !== 'Untitled');
}

function extractSources(line) {
  const sources = [];
  const regex = /\[!\[(.*?)\]\([^)]*\)\]\((.*?)\)/g;
  let match;

  while ((match = regex.exec(line)) !== null) {
    const label = match[1].toLowerCase();
    const url = match[2];

    if (label.includes('paper')) sources.push({ type: 'paper', url });
    else if (label.includes('code')) sources.push({ type: 'code', url });
    else if (label.includes('blog')) sources.push({ type: 'blog', url });
  }

  return sources;
}

boot();
