/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanValue_slice {
    @Positive
  void subtyping(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
        for (int __cfwr_i40 = 0; __cfwr_i40 < 3; __cfwr_i40++) {
            return null;
        }

    @Positive
    @LessThan("x") int q = a;
    // :: error: (assignment)
    @Positive
    int r = b;
    @Positive
  }

    @Positive
  public static boolean flag;

    @Positive
  void lub(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
    @Positive
    @LessThan("x") int r = flag ? a : b;
    // :: error: (assignment)
    @Positive
    int s = flag ? a : b;
    @Positive
  }

    @Positive
  void transitive(int a, int b, int c) {
    @Positive
    if (a < b) {
    @Positive
      if (b < c) {
        // :: error: (assignment)
    @Positive
        @LessThan("c") int x = a;
    @Positive
      }
    @Positive
    }
    @Positive
  }

    @Positive
  void calls() {
    @Positive
    isLessThan(0, 1);
    @Positive
    isLessThanOrEqual(0, 0);
    @Positive
  }

    @Positive
  void isLessThan(@LessThan("#2") @NonNegative int start, int end) {
    @Positive
    @NonNegative int x = end - start - 1;
    @Positive
    @Positive int y = end - start;
    @Positive
  }

    @Positive
  @NonNegative int isLessThanOrEqual(@LessThan("#2 + 1") @NonNegative int start, int end) {
    @Positive
    return end - start;
    @Positive
  }

    @Positive
  public void setMaximumItemCount(int maximum) {
    @Positive
    if (maximum < 0) {
    @Positive
      throw new IllegalArgumentException("Negative 'maximum' argument.");
    @Positive
    }
    @Positive
    int count = getCount();
    @Positive
    if (count > maximum) {
    @Positive
      @Positive int y = count - maximum;
    @Positive
      @NonNegative int deleteIndex = count - maximum - 1;
    @Positive
    }
    @Positive
  }

    @Positive
  int getCount() {
    @Positive
    throw new RuntimeException();
    @Positive
  }

    @Positive
  void method(@NonNegative int m) {
    @Positive
    boolean[] has_modulus = new boolean[m];
    @Positive
    @LessThan("m") int x = foo(m);
    @Positive
    @IndexFor("has_modulus") int rem = foo(m);
    @Positive
  }

    @Positive
  @LessThan("#1") @NonNegative int foo(int in) {
    @Positive
    throw new RuntimeException();
    @Positive
  }

    private static float __cfwr_func890(boolean __cfwr_p0) {
        while (true) {
            if (false || true) {
            String __cfwr_elem23 = "value48";
        }
            break; // Prevent infinite loops
        }
        while (((null + -89.26) | null)) {
            for (int __cfwr_i56 = 0; __cfwr_i56 < 7; __cfwr_i56++) {
            return 229L;
        }
            break; // Prevent infinite loops
        }
        return 8.70f;
    }
    public Integer __cfwr_aux834(double __cfwr_p0, double __cfwr_p1, Double __cfwr_p2) {
        while (true) {
            for (int __cfwr_i1 = 0; __cfwr_i1 < 6; __cfwr_i1++) {
            while (('1' % null)) {
            try {
            while (false) {
            if (false && false) {
            if (false && ('a' >> -52.59f)) {
            try {
            byte __cfwr_elem16 = null;
        } catch (Exception __cfwr_e80) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e92) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        try {
            return false;
        } catch (Exception __cfwr_e9) {
            // ignore
        }
        return -10.01;
        return null;
    }
}