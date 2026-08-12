Replace GridSearchCV in examples/word_language_model/train.py with bounded 1-D search over learning rate with explicit bounds (lower, upper) and iteration cap. Keep single-config training under __main__ guard, module import-safe. Search logic must be importable as optimizer class constructed with bounds.

CLI must support --search-iter (default DEFAULT_SEARCH_ITERATIONS=16) for iteration cap, --lr-low (default 1.0) and --lr-high (default 40.0) for bounds, plus existing --seed, --data, --bptt, --batch_size, --epochs, --data-limit, --save, --no-cuda — all under if __name__ == '__main__' / main() guard, import-safe.

Training wiring must preserve original skorch setup: define my_train_split(ds, y) returning (ds, skorch.dataset.Dataset(corpus.valid[:200], y=None)), corpus = data.Corpus(args.data), ntokens = len(corpus.dictionary), torch.manual_seed(args.seed), device = 'cuda' if args.cuda else 'cpu', Net with train_split=my_train_split, iterator_train=data.Loader with iterator_train__device=device and iterator_train__bptt=args.bptt, iterator_valid=data.Loader with iterator_valid__device=device and iterator_valid__bptt=args.bptt, X = corpus.train[:args.data_limit].numpy() handling -1 as full data, callbacks [Checkpoint, ProgressBar, LRAnnealing] preserving original, LRAnnealing defined inside main.

Objective must do net.set_params(lr=float(x)) and evaluate via sklearn.model_selection.cross_val_score averaged over X (import cross_val_score inside main or top), handle data-limit via slicing + .numpy().

After search, set best lr, net.fit(X), net.save_params(f_params=args.save), print best lr and evaluation count. Define DEFAULT_SEARCH_ITERATIONS = 16 constant and remove GridSearchCV import/instantiation.

Optimizer trust constants: N_SEED=3 seed points, INIT_RADIUS_FRAC=0.25 of bound width, SHRINK=0.5 on non-improvement, RNG np.random.default_rng(seed).uniform(lo,hi) — same seed gives same result.
