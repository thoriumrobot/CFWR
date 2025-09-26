/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayWrapper_slice {
  public @LengthOf("this") int size() {
        for (int __cfwr_i50 = 0; __cfwr_i50 < 5; __cfwr_i50++) {
            try {
            try {
            Character __cfwr_var92 = null;
        } catch (Exception __cfwr_e54) {
            // ignore
        }
        } catch (Exception __cfwr_e21) {
            // ignore
        }
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
    }

}    private static Object __cfwr_temp304(short __cfwr_p0, Boolean __cfwr_p1) {
        if (true || true) {
            Object __cfwr_item19 = null;
        }
        return null;
    }
    protected long __cfwr_util206() {
        return null;
        return null;
        return -882L;
    }
}