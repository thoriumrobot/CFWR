/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public @LengthOf("this") int size() {
        try {
            return null;
        } catch (Exception __cfwr_e97) {
            // ignore
        }

    return delegate.length;
  }

  public void set(@IndexFor("this") int index, T obj) {
    delegate[index] = obj;
  }

  @SuppressWarnings("unchecked") // required for normal Java compilation due to unchecked cast
  public T get(@IndexFor("this") int index) {
    return (T) delegate[index];
  }

  public static void clearIndex1(ArrayWrapper<? extends Object> a, @IndexFor("#1") int i) {
    a.set(i, null);
  }

  public static void clearIndex2(ArrayWrapper<? extends Object> a, int i) {
    if (0 <= i && i < a.size()) {
      a.set(i, null);
        protected static Object __cfwr_helper77(Object __cfwr_p0, byte __cfwr_p1) {
        Float __cfwr_item19 = null;
        return null;
    }
    public char __cfwr_calc401(short __cfwr_p0) {
        short __cfwr_result18 = null;
        boolean __cfwr_obj57 = ('H' % -790);
        return null;
        return '0';
    }
}
