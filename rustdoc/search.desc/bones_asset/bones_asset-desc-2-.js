searchState.loadedDescShard("bones_asset", 2, "Creates an empty reader.\nReturns the contents of the internal buffer, filling it …\nFlushes the stream to ensure that all buffered contents …\nThe actual reader/writer we are wrapping.\nReturns a stream over the lines of this byte stream.\nAttempt to close the object.\nAttempt to return the contents of the internal buffer, …\nAttempt to flush the object, ensuring that any buffered …\nAttempt to read from the <code>AsyncRead</code> into <code>buf</code>.\nAttempt to read from the <code>AsyncRead</code> into <code>bufs</code> using vectored\nAttempt to seek to an offset, in bytes, in a stream.\nAttempt to write bytes from <code>buf</code> into the object.\nAttempt to write bytes from <code>bufs</code> into the object using …\nReads some bytes from the byte stream.\nReads the exact number of bytes required to fill <code>buf</code>.\nReads all bytes and appends them into <code>buf</code> until a newline …\nReads the entire contents and appends them to a <code>Vec</code>.\nReads the entire contents and appends them to a <code>String</code>.\nReads all bytes and appends them into <code>buf</code> until the …\nLike <code>read()</code>, except it reads into a slice of buffers.\nCreates an infinite reader that reads the same byte …\nSeeks to a new position in a byte stream.\nCreates a writer that consumes and drops all data.\nSplits a stream into <code>AsyncRead</code> and <code>AsyncWrite</code> halves.\nReturns a stream over the contents of this reader split on …\nCreates an adapter which will read at most <code>limit</code> bytes …\nWrites some bytes into the byte stream.\nWrites an entire buffer into the byte stream.\nLike <code>write()</code>, except that it writes a slice of buffers.\nRead bytes asynchronously.\nRead bytes asynchronously.\nSeek bytes asynchronously.\nWrite bytes asynchronously.\nA future represents an asynchronous computation obtained …\nValues yielded by the stream.\nThe type of value produced on completion.\nA stream of values produced asynchronously.\nExtension trait for <code>AsyncWrite</code>.\nExtension trait for <code>Future</code>.\nExtension trait for <code>Stream</code>.\nExtension trait for <code>AsyncBufRead</code>.\nExtension trait for <code>AsyncSeek</code>.\nExtension trait for <code>AsyncRead</code>.\nTests if <code>predicate</code> returns <code>true</code> for all items in the …\nTests if <code>predicate</code> returns <code>true</code> for any item in the stream.\nBoxes the future and changes its type to …\nBoxes the stream and changes its type to …\nBoxes the future and changes its type to <code>dyn Future + &#39;a</code>.\nBoxes the stream and changes its type to <code>dyn Stream + &#39;a</code>.\nBoxes the reader and changes its type to …\nBoxes the writer and changes its type to …\nConverts this <code>AsyncRead</code> into a <code>Stream</code> of bytes.\nCatches panics while polling the future.\nAppends another stream to the end of this one.\nCreates an adapter which will chain this stream with …\nClones all items.\nCloses the writer.\nCollects all items in the stream into a collection.\nTells this buffer that <code>amt</code> bytes have been consumed from …\nConsumes <code>amt</code> buffered bytes.\nCopies all items.\nCounts the number of items in the stream.\nRepeats the stream from beginning to end, forever.\nYields all immediately available values from a stream.\nEnumerates items, mapping them to <code>(index, item)</code>.\nReturns the contents of the internal buffer, filling it …\nKeeps items of the stream for which <code>predicate</code> returns <code>true</code>.\nFilters and maps items of the stream using a closure.\nFinds the first item of the stream for which <code>predicate</code> …\nApplies a closure to items in the stream and returns the …\nMaps items to streams and then concatenates them.\nConcatenates inner streams.\nFlushes the stream to ensure that all buffered contents …\nAccumulates a computation over the stream.\nCalls a closure on each item of the stream.\nFuses the stream so that it stops yielding items after the …\nCalls a closure on each item and passes it on.\nReturns the last item in the stream.\nReturns a stream over the lines of this byte stream.\nMaps items of the stream to new values using a closure.\nRetrieves the next item in the stream.\nGets the <code>n</code>th item of the stream.\nReturns the result of <code>self</code> or <code>other</code> future, preferring <code>self</code>…\nMerges with <code>other</code> stream, preferring items from <code>self</code> …\nPartitions items into those for which <code>predicate</code> is <code>true</code> …\nAttempt to resolve the future to a final value, registering\nA convenience for calling <code>Future::poll()</code> on <code>!</code><code>Unpin</code> types.\nAttempt to close the object.\nAttempt to return the contents of the internal buffer, …\nAttempt to flush the object, ensuring that any buffered …\nAttempt to pull out the next value of this stream, …\nA convenience for calling <code>Stream::poll_next()</code> on <code>!</code><code>Unpin</code> …\nAttempt to read from the <code>AsyncRead</code> into <code>buf</code>.\nAttempt to read from the <code>AsyncRead</code> into <code>bufs</code> using vectored\nAttempt to seek to an offset, in bytes, in a stream.\nAttempt to write bytes from <code>buf</code> into the object.\nAttempt to write bytes from <code>bufs</code> into the object using …\nFinds the index of the first item of the stream for which …\nReturns the result of <code>self</code> or <code>other</code> future, with no …\nMerges with <code>other</code> stream, with no preference for either …\nReads some bytes from the byte stream.\nReads the exact number of bytes required to fill <code>buf</code>.\nReads all bytes and appends them into <code>buf</code> until a newline …\nReads the entire contents and appends them to a <code>Vec</code>.\nReads the entire contents and appends them to a <code>String</code>.\nReads all bytes and appends them into <code>buf</code> until the …\nLike <code>read()</code>, except it reads into a slice of buffers.\nMaps items of the stream to new values using a state value …\nSeeks to a new position in a byte stream.\nReturns the bounds on the remaining length of the stream.\nSkips the first <code>n</code> items of the stream.\nSkips items while <code>predicate</code> returns <code>true</code>.\nReturns a stream over the contents of this reader split on …\nYields every <code>step</code>th item.\nTakes only the first <code>n</code> items of the stream.\nCreates an adapter which will read at most <code>limit</code> bytes …\nTakes items while <code>predicate</code> returns <code>true</code>.\nMaps items of the stream to new values using an async …\nCollects all items in the fallible stream into a …\nAccumulates a fallible computation over the stream.\nCalls a fallible closure on each item of the stream, …\nRetrieves the next item in the stream.\nCollects a stream of pairs into a pair of collections.\nWrites some bytes into the byte stream.\nWrites an entire buffer into the byte stream.\nLike <code>write()</code>, except that it writes a slice of buffers.\nZips up two streams into a single stream of pairs.\nFuture for the <code>StreamExt::all()</code> method.\nFuture for the <code>StreamExt::any()</code> method.\nIterator for the <code>block_on()</code> function.\nType alias for …\nType alias for <code>Pin&lt;Box&lt;dyn Stream&lt;Item = T&gt; + &#39;static&gt;&gt;</code>.\nStream for the <code>StreamExt::chain()</code> method.\nStream for the <code>StreamExt::cloned()</code> method.\nFuture for the <code>StreamExt::collect()</code> method.\nStream for the <code>StreamExt::copied()</code> method.\nFuture for the <code>StreamExt::count()</code> method.\nStream for the <code>StreamExt::cycle()</code> method.\nStream for the <code>StreamExt::drain()</code> method.\nStream for the <code>empty()</code> function.\nStream for the <code>StreamExt::enumerate()</code> method.\nStream for the <code>StreamExt::filter()</code> method.\nStream for the <code>StreamExt::filter_map()</code> method.\nFuture for the <code>StreamExt::find()</code> method.\nFuture for the <code>StreamExt::find_map()</code> method.\nStream for the <code>StreamExt::flat_map()</code> method.\nStream for the <code>StreamExt::flatten()</code> method.\nFuture for the <code>StreamExt::fold()</code> method.\nFuture for the <code>StreamExt::for_each()</code> method.\nStream for the <code>StreamExt::fuse()</code> method.\nStream for the <code>StreamExt::inspect()</code> method.\nValues yielded by the stream.\nStream for the <code>iter()</code> function.\nFuture for the <code>StreamExt::last()</code> method.\nStream for the <code>StreamExt::map()</code> method.\nFuture for the <code>StreamExt::next()</code> method.\nFuture for the <code>StreamExt::nth()</code> method.\nStream for the <code>once()</code> function.\nStream for the <code>once_future()</code> method.\nStream for the <code>or()</code> function and the <code>StreamExt::or()</code> …\nFuture for the <code>StreamExt::partition()</code> method.\nStream for the <code>pending()</code> function.\nStream for the <code>poll_fn()</code> function.\nFuture for the <code>StreamExt::position()</code> method.\nStream for the <code>race()</code> function and the <code>StreamExt::race()</code> …\nStream for the <code>repeat()</code> function.\nStream for the <code>repeat_with()</code> function.\nStream for the <code>StreamExt::scan()</code> method.\nStream for the <code>StreamExt::skip()</code> method.\nStream for the <code>StreamExt::skip_while()</code> method.\nStream for the <code>StreamExt::step_by()</code> method.\nA stream of values produced asynchronously.\nExtension trait for <code>Stream</code>.\nStream for the <code>StreamExt::take()</code> method.\nStream for the <code>StreamExt::take_while()</code> method.\nStream for the <code>StreamExt::then()</code> method.\nFuture for the <code>StreamExt::try_collect()</code> method.\nFuture for the <code>StreamExt::try_fold()</code> method.\nFuture for the <code>StreamExt::try_for_each()</code> method.\nFuture for the <code>StreamExt::try_next()</code> method.\nStream for the <code>try_unfold()</code> function.\nStream for the <code>unfold()</code> function.\nFuture for the <code>StreamExt::unzip()</code> method.\nStream for the <code>StreamExt::zip()</code> method.\nTests if <code>predicate</code> returns <code>true</code> for all items in the …\nTests if <code>predicate</code> returns <code>true</code> for any item in the stream.\nConverts a stream into a blocking iterator.\nBoxes the stream and changes its type to …\nBoxes the stream and changes its type to <code>dyn Stream + &#39;a</code>.\nAppends another stream to the end of this one.\nClones all items.\nCollects all items in the stream into a collection.\nCopies all items.\nCounts the number of items in the stream.\nRepeats the stream from beginning to end, forever.\nYields all immediately available values from a stream.\nCreates an empty stream.\nEnumerates items, mapping them to <code>(index, item)</code>.\nKeeps items of the stream for which <code>predicate</code> returns <code>true</code>.\nFilters and maps items of the stream using a closure.\nFinds the first item of the stream for which <code>predicate</code> …\nApplies a closure to items in the stream and returns the …\nMaps items to streams and then concatenates them.\nConcatenates inner streams.\nAccumulates a computation over the stream.\nCalls a closure on each item of the stream.\nFuses the stream so that it stops yielding items after the …\nCalls a closure on each item and passes it on.\nCreates a stream from an iterator.\nReturns the last item in the stream.\nMaps items of the stream to new values using a closure.\nRetrieves the next item in the stream.\nGets the <code>n</code>th item of the stream.\nCreates a stream that yields a single item.\nCreates a stream that invokes the given future as its …\nMerges two streams, preferring items from <code>stream1</code> whenever …\nMerges with <code>other</code> stream, preferring items from <code>self</code> …\nPartitions items into those for which <code>predicate</code> is <code>true</code> …\nCreates a stream that is always pending.\nCreates a stream from a function returning <code>Poll</code>.\nAttempt to pull out the next value of this stream, …\nA convenience for calling <code>Stream::poll_next()</code> on <code>!</code><code>Unpin</code> …\nFinds the index of the first item of the stream for which …\nMerges two streams, with no preference for either stream …\nMerges with <code>other</code> stream, with no preference for either …\nRaces two streams, but with a user-provided seed for …\nCreates an infinite stream that yields the same item …\nCreates an infinite stream from a closure that generates …\nMaps items of the stream to new values using a state value …\nReturns the bounds on the remaining length of the stream.\nSkips the first <code>n</code> items of the stream.\nSkips items while <code>predicate</code> returns <code>true</code>.\nYields every <code>step</code>th item.\nTakes only the first <code>n</code> items of the stream.\nTakes items while <code>predicate</code> returns <code>true</code>.\nMaps items of the stream to new values using an async …\nCollects all items in the fallible stream into a …\nAccumulates a fallible computation over the stream.\nCalls a fallible closure on each item of the stream, …\nRetrieves the next item in the stream.\nCreates a stream from a seed value and a fallible async …\nCreates a stream from a seed value and an async closure …\nCollects a stream of pairs into a pair of collections.\nZips up two streams into a single stream of pairs.\nA builder for default Fx hashers.\nA <code>HashMap</code> using a default Fx hasher.\nA <code>HashSet</code> using a default Fx hasher.\nThis hashing algorithm was extracted from the Rustc …\nThis hashing algorithm was extracted from the Rustc …\nThis hashing algorithm was extracted from the Rustc …\nA convenience function for when you need a quick usize …\nA convenience function for when you need a quick 32-bit …\nA convenience function for when you need a quick 64-bit …\nThe memory allocator returned an error\nError due to the computed capacity exceeding the collection…\nKey equivalence trait.\nA hash map implemented with quadratic probing and SIMD …\nA hash set implemented as a <code>HashMap</code> where the value is <code>()</code>.\nLow-level hash table with explicit hashing.\nThe error type for <code>try_reserve</code> methods.\nChecks if this value is equivalent to the given key.\nA hash map implemented with quadratic probing and SIMD …\nA hash set implemented as a <code>HashMap</code> where the value is <code>()</code>.\nA hash table implemented with quadratic probing and SIMD …\nExperimental and unsafe <code>RawTable</code> API. This module is only …\nThe layout of the allocation request that failed.\nDefault hasher for <code>HashMap</code>.\nA draining iterator over the entries of a <code>HashMap</code> in …\nA view into a single entry in a map, which may either be …\nA view into a single entry in a map, which may either be …\nA draining iterator over entries of a <code>HashMap</code> which don’…\nA hash map implemented with quadratic probing and SIMD …\nAn owning iterator over the entries of a <code>HashMap</code> in …\nAn owning iterator over the keys of a <code>HashMap</code> in arbitrary …\nAn owning iterator over the values of a <code>HashMap</code> in …\nAn iterator over the entries of a <code>HashMap</code> in arbitrary …\nA mutable iterator over the entries of a <code>HashMap</code> in …\nAn iterator over the keys of a <code>HashMap</code> in arbitrary order. …\nAn occupied entry.\nAn occupied entry.\nAn occupied entry.\nA view into an occupied entry in a <code>HashMap</code>. It is part of …\nA view into an occupied entry in a <code>HashMap</code>. It is part of …\nThe error returned by <code>try_insert</code> when the key already …\nA builder for computing where in a <code>HashMap</code> a key-value …\nA builder for computing where in a <code>HashMap</code> a key-value …\nA view into a single entry in a map, which may either be …\nA view into an occupied entry in a <code>HashMap</code>. It is part of …\nA view into a vacant entry in a <code>HashMap</code>. It is part of the …\nA vacant entry.\nA vacant entry.\nA vacant entry.\nA view into a vacant entry in a <code>HashMap</code>. It is part of the …\nA view into a vacant entry in a <code>HashMap</code>. It is part of the …\nAn iterator over the values of a <code>HashMap</code> in arbitrary …\nA mutable iterator over the values of a <code>HashMap</code> in …\nThe entry in the map that was already occupied.\nThe value which was not inserted, because the entry was …\nA lazy iterator producing elements in the difference of …\nA draining iterator over the items of a <code>HashSet</code>.\nA view into a single entry in a set, which may either be …\nA draining iterator over entries of a <code>HashSet</code> which don’…\nA hash set implemented as a <code>HashMap</code> where the value is <code>()</code>.\nA lazy iterator producing elements in the intersection of …\nAn owning iterator over the items of a <code>HashSet</code>.\nAn iterator over the items of a <code>HashSet</code>.\nAn occupied entry.\nA view into an occupied entry in a <code>HashSet</code>. It is part of …\nA lazy iterator producing elements in the symmetric …\nA lazy iterator producing elements in the union of <code>HashSet</code>…\nA vacant entry.\nA view into a vacant entry in a <code>HashSet</code>. It is part of the …\nType representing the absence of an entry, as returned by …\nA draining iterator over the items of a <code>HashTable</code>.\nA view into a single entry in a table, which may either be …\nA draining iterator over entries of a <code>HashTable</code> which don…\nLow-level hash table with explicit hashing.\nAn owning iterator over the entries of a <code>HashTable</code> in …\nAn iterator over the entries of a <code>HashTable</code> in arbitrary …\nA mutable iterator over the entries of a <code>HashTable</code> in …\nAn occupied entry.\nA view into an occupied entry in a <code>HashTable</code>. It is part …\nA vacant entry.\nA view into a vacant entry in a <code>HashTable</code>. It is part of …\nA reference to a hash table bucket containing a <code>T</code>.\nA reference to an empty bucket into which an can be …\nIterator which consumes elements without freeing the table …\nIterator which consumes a table and returns elements.\nIterator which returns a raw pointer to every full bucket …\nIterator over occupied buckets that could match a given …\nA raw hash table with an unsafe API.\nA Condition Variable\nA closure has completed successfully.\nA mutual exclusive primitive that is always fair, useful …\nAn RAII implementation of a “scoped lock” of a mutex. …\nA thread is currently executing a closure.\nAn RAII mutex guard returned by <code>FairMutexGuard::map</code>, which …\nAn RAII mutex guard returned by <code>MutexGuard::map</code>, which can …\nAn RAII mutex guard returned by <code>ReentrantMutexGuard::map</code>, …\nAn RAII read lock guard returned by <code>RwLockReadGuard::map</code>, …\nAn RAII write lock guard returned by <code>RwLockWriteGuard::map</code>…\nA mutual exclusion primitive useful for protecting shared …\nAn RAII implementation of a “scoped lock” of a mutex. …\nA closure has not been executed yet\nA synchronization primitive which can be used to run a …\nCurrent state of a <code>Once</code>.\nA closure was executed but panicked.\nRaw fair mutex type backed by the parking lot.\nRaw mutex type backed by the parking lot.\nRaw reader-writer lock type backed by the parking lot.\nImplementation of the <code>GetThreadId</code> trait for …\nA mutex which can be recursively locked by a single thread.\nAn RAII implementation of a “scoped lock” of a …\nA reader-writer lock\nRAII structure used to release the shared read access of a …\nRAII structure used to release the upgradable read access …\nRAII structure used to release the exclusive write access …\nA type indicating whether a timed wait on a condition …\nCreates a new fair mutex in an unlocked state ready for …\nCreates a new mutex in an unlocked state ready for use.\nCreates a new reentrant mutex in an unlocked state ready …\nCreates a new instance of an <code>RwLock&lt;T&gt;</code> which is unlocked.\nDuration type used for <code>try_lock_for</code>.\nDuration type used for <code>try_lock_for</code>.\nHelper trait which returns a non-zero thread ID.\nMarker type which determines whether a lock guard should …\nMarker type which determines whether a lock guard should …\nMarker type which indicates that the Guard type for a lock …\nMarker type which indicates that the Guard type for a lock …\nInitial value for an unlocked mutex.\nInitial value.\nInitial value for an unlocked <code>RwLock</code>.\nInstant type used for <code>try_lock_until</code>.\nInstant type used for <code>try_lock_until</code>.\nAn RAII mutex guard returned by <code>MutexGuard::map</code>, which can …\nAn RAII mutex guard returned by <code>ReentrantMutexGuard::map</code>, …\nAn RAII read lock guard returned by <code>RwLockReadGuard::map</code>, …\nAn RAII write lock guard returned by <code>RwLockWriteGuard::map</code>…\nA mutual exclusion primitive useful for protecting shared …\nAn RAII implementation of a “scoped lock” of a mutex. …\nBasic operations for a mutex.\nAdditional methods for mutexes which support fair …\nAdditional methods for mutexes which support locking with …\nA raw mutex type that wraps another raw mutex to provide …\nBasic operations for a reader-writer lock.\nAdditional methods for <code>RwLock</code>s which support atomically …\nAdditional methods for <code>RwLock</code>s which support fair …\nAdditional methods for <code>RwLock</code>s which support recursive …\nAdditional methods for <code>RwLock</code>s which support recursive …\nAdditional methods for <code>RwLock</code>s which support locking with …\nAdditional methods for <code>RwLock</code>s which support atomically …\nAdditional methods for <code>RwLock</code>s which support upgradable …\nAdditional methods for <code>RwLock</code>s which support upgradable …\nAdditional methods for <code>RwLock</code>s which support upgradable …\nA mutex which can be recursively locked by a single thread.\nAn RAII implementation of a “scoped lock” of a …\nA reader-writer lock\nRAII structure used to release the shared read access of a …\nRAII structure used to release the upgradable read access …\nRAII structure used to release the exclusive write access …\nTemporarily yields the mutex to a waiting thread if there …\nTemporarily yields an exclusive lock to a waiting thread …\nTemporarily yields a shared lock to a waiting thread if …\nTemporarily yields an upgradable lock to a waiting thread …\nAtomically downgrades an exclusive lock into a shared lock …\nDowngrades an exclusive lock to an upgradable lock.\nDowngrades an upgradable lock to a shared lock.\nChecks whether the mutex is currently locked.\nChecks if this <code>RwLock</code> is currently locked in any way.\nCheck if this <code>RwLock</code> is currently exclusively locked.\nAcquires this mutex, blocking the current thread until it …\nAcquires an exclusive lock, blocking the current thread …\nAcquires a shared lock, blocking the current thread until …\nAcquires a shared lock without deadlocking in case of a …\nAcquires an upgradable lock, blocking the current thread …\nReturns a non-zero thread ID which identifies the current …\nAttempts to acquire this mutex without blocking. Returns …\nAttempts to acquire an exclusive lock without blocking.\nAttempts to acquire an exclusive lock until a timeout is …\nAttempts to acquire an exclusive lock until a timeout is …\nAttempts to acquire this lock until a timeout is reached.\nAttempts to acquire a shared lock without blocking.\nAttempts to acquire a shared lock until a timeout is …\nAttempts to acquire a shared lock without deadlocking in …\nAttempts to acquire a shared lock until a timeout is …\nAttempts to acquire a shared lock until a timeout is …\nAttempts to acquire a shared lock until a timeout is …\nAttempts to acquire this lock until a timeout is reached.\nAttempts to acquire an upgradable lock without blocking.\nAttempts to acquire an upgradable lock until a timeout is …\nAttempts to acquire an upgradable lock until a timeout is …\nAttempts to upgrade an upgradable lock to an exclusive …\nAttempts to upgrade an upgradable lock to an exclusive …\nAttempts to upgrade an upgradable lock to an exclusive …\nUnlocks this mutex.\nReleases an exclusive lock.\nReleases an exclusive lock using a fair unlock protocol.\nUnlocks this mutex using a fair unlock protocol.\nReleases a shared lock.\nReleases a shared lock using a fair unlock protocol.\nReleases an upgradable lock.\nReleases an upgradable lock using a fair unlock protocol.\nUpgrades an upgradable lock to an exclusive lock.\nVariant for fast PRNGs, like Wyrand.\nTrait for enabling creating new <code>TurboCore</code> instances from …\nDetermines the kind of PRNG. <code>TurboKind::FAST</code> RNGs are …\nThis trait provides the means to easily generate all …\nA Random Number generator, powered by the <code>WyRand</code> algorithm.\nVariant for slower PRNGs, like ChaCha8.\nA marker trait to be applied to anything that implements …\nAssociated type for accepting valid Seed values. Must be …\nTrait for implementing Seedable PRNGs, requiring that the …\nBase trait for implementing a PRNG. Only one method must be\nEnum for determining the kind of PRNG, whether a fast one, …\nExtension trait for automatically implementing all …\nGenerates a random <code>char</code> in ranges a-z and A-Z.\nGenerates a random <code>char</code> in ranges a-z, A-Z and 0-9.\nReturns a random boolean value.\nReturns a boolean value based on a rate. <code>rate</code> represents …\nGenerates a random <code>char</code> in the given range.\nGenerate a random digit in the given <code>radix</code>.\nReturns a random <code>f32</code> value between <code>0.0</code> and <code>1.0</code>.\nReturns a random <code>f32</code> value between <code>-1.0</code> and <code>1.0</code>.\nReturns a random <code>f32</code> value between <code>0.0</code> and <code>1.0</code>.\nReturns a random <code>f32</code> value between <code>-1.0</code> and <code>1.0</code>.\nFills a mutable buffer with random bytes.\nForks a <code>TurboCore</code> instance by deterministically deriving a …\nReturns an array of constant <code>SIZE</code> containing random <code>u8</code> …\nReturns a random <code>i128</code> value.\nReturns a random <code>i16</code> value.\nReturns a random <code>i32</code> value.\nReturns a random <code>i64</code> value.\nReturns a random <code>i8</code> value.\nReturns a random <code>isize</code> value.\nReturns a random <code>u128</code> value.\nReturns a random <code>u16</code> value.\nReturns a random <code>u32</code> value.\nReturns a random <code>u64</code> value.\nReturns a random <code>u8</code> value.\nReturns a random <code>usize</code> value.\nReturns a random <code>i128</code> within a given range bound.\nReturns a random <code>i16</code> value.\nReturns a random <code>i32</code> value.\nReturns a random <code>i64</code> value.\nReturns a random <code>i8</code> value.\nReturns a <code>usize</code> value for stable indexing across different …\nReturns a random <code>isize</code> within a given range bound.\nGenerates a random <code>char</code> in the range a-z.\nPartially shuffles a slice by a given amount and returns …\nReseeds the <code>SeededCore</code> with a new seed/state.\nSamples a random item from a slice of values.\nSamples a random item from an iterator of values. <code>O(1)</code> if …\nSamples multiple unique items from a slice of values.\nSamples multiple unique items from an iterator of values.\nSamples multiple unique items from a mutable slice of …\nSamples a random <code>&amp;mut</code> item from a slice of values.\nShuffles a slice randomly in O(n) time.\nReturns a random <code>u128</code> within a given range bound.\nReturns a random <code>u16</code> value.\nReturns a random <code>u32</code> value.\nReturns a random <code>u64</code> value.\nReturns a random <code>u8</code> value.\nGenerates a random <code>char</code> in the range A-Z.\nReturns a random <code>usize</code> within a given range bound.\nStochastic Acceptance implementation of Roulette Wheel …\nStochastic Acceptance implementation of Roulette Wheel …\nCreates a new <code>SeededCore</code> with a specific seed value.\nA Random Number generator, powered by the <code>WyRand</code> algorithm.")