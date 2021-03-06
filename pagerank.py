import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000
CONVERGENCE = 0.001


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    outgoing_links = corpus[page]

    # Initialize with equal probability
    probability_distro = {
        corpus_page: float(1 / len(corpus)) for corpus_page in corpus.keys()
    }

    # If no outgoing links, return the probability distribution
    if len(outgoing_links) == 0:
        return probability_distro

    damping_probability = float(damping_factor / len(outgoing_links))
    random_probability = float((1 - damping_factor) / len(corpus))

    # Reassign the probability per page in the probability distribution
    for pd_page in probability_distro.keys():
        if pd_page in outgoing_links:
            probability_distro[pd_page] = damping_probability + random_probability
        else:
            probability_distro[pd_page] = random_probability

    return probability_distro


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Keep track of page hits to calculate pagerank
    page_hits = {corpus_page: 0 for corpus_page in corpus.keys()}
    current_page = None
    current_n = 0

    while current_n < n:
        # Start by choosing a random page of equal probability
        if current_page is None:
            current_page = random.choice(list(corpus))
            page_hits[current_page] += 1
            current_n += 1
            continue
        model = transition_model(corpus, current_page, damping_factor)
        # Choose a page given the probability as weights
        current_page = random.choices(list(model.keys()), list(model.values()))[0]
        page_hits[current_page] += 1
        current_n += 1

    # Determine pagerank by dividing hits by the sample number
    return {page: hits / n for page, hits in page_hits.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Initialize every pagerank with equal probability
    pageranks = {page: float(1 / len(corpus)) for page in corpus.keys()}

    return rank_pages(corpus, damping_factor, pageranks)


def rank_pages(
    corpus: dict[str, set[str]], damping_factor: float, pageranks: dict[str, float]
) -> dict[str, float]:
    """Determine new page ranks from given page ranks."""

    # Keep track to compare for convergence
    old_pageranks = pageranks.copy()

    for page_p, links in corpus.items():
        if len(links) > 0:
            pageranks[page_p] = (1 - damping_factor) / len(
                corpus
            ) + damping_factor * sum(
                [
                    pageranks[link] / len(corpus[link])
                    for link in [
                        page_i
                        for page_i, page_i_links in corpus.items()
                        if page_p in page_i_links
                    ]  # Creates an array of pages (i) that link to the targeted page (p)
                ]
            )
        else:
            corpus[page_p] = set(corpus.keys())
            pageranks[page_p] = (1 - damping_factor) / len(
                corpus
            ) + damping_factor * sum(
                [pageranks[link] / len(corpus[link]) for link in links]
            )  #  If no links, choose from every page (inclusive) instead

    converged = True
    for page, pagerank in pageranks.items():
        if abs(old_pageranks[page] - pagerank) > CONVERGENCE:
            converged = False
            break

    if not converged:
        rank_pages(corpus, damping_factor, pageranks)

    return pageranks


if __name__ == "__main__":
    main()
