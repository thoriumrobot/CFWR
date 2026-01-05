/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ParserOffsetTest_slice {
    @Positive
  public void addition3(String[] a, @IndexFor("#1") int i) {
        char __cfwr_var43 = 'c';

    @Positive
    if ((i + 5) < a.length) {
    @Positive
      @IndexFor("a") int j = i + 5;
    @Positive
    }
    @Positive
  }

    @Positive
  public void subtraction3(String[] a, @NonNegative int k) {
    @Positive
    if (k - 5 < a.length) {
    @Positive
      String s = a[k - 5];
    @Positive
      @IndexFor("a") int j = k - 5;
    @Positive
    }
    @Positive
  }

    @Positive
  public void subtraction4(String[] a, @IndexFor("#1") int i) {
    @Positive
    if (1 - i < a.length) {
      // The error on this assignment is a false positive.
      // :: error: (assignment)
    @Positive
      @IndexFor("a") int j = 1 - i;

      // :: error: (assignment)
    @Positive
      @LTLengthOf(value = "a", offset = "1") int k = i;
    @Positive
    }
    @Positive
  }

    @Positive
  public void subtraction5(String[] a, int i) {
    @Positive
    if (1 - i < a.length) {
      // :: error: (assignment)
    @Positive
      @IndexFor("a") int j = i;
    @Positive
    }
    @Positive
  }

    public static Long __cfwr_util182(double __cfwr_p0, Character __cfwr_p1) {
        try {
            if (true || false) {
            if (false || false) {
            return -14.35f;
        }
        }
        } catch (Exception __cfwr_e98) {
            // ignore
        }
        Character __cfwr_data39 = null;
        return null;
    }
    static Integer __cfwr_temp219() {
        while ((65.78 | null)) {
            if ((277L << -90.55f) || false) {
            try {
            try {
            try {
            for (int __cfwr_i42 = 0; __cfwr_i42 < 3; __cfwr_i42++) {
            if (false && true) {
            return null;
        }
        }
        } catch (Exception __cfwr_e18) {
            // ignore
        }
        } catch (Exception __cfwr_e40) {
            // ignore
        }
        } catch (Exception __cfwr_e56) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        return null;
        for (int __cfwr_i63 = 0; __cfwr_i63 < 5; __cfwr_i63++) {
            while ((813 >> -63.78)) {
            byte __cfwr_elem8 = (null >> null);
            break; // Prevent infinite loops
        }
        }
        return null;
    }
}