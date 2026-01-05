/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanValue_slice {
    @Positive
  void subtyping(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
        try {
            try {
            while ((false >> 'r')) {
            try {
            boolean __cfwr_result37 = (25.57 * -4);
        } catch (Exception __cfwr_e58) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e86) {
            // ignore
        }
        } catch (Exception __cfwr_e45) {
            // ignore
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

    public long __cfwr_func857(String __cfwr_p0, short __cfwr_p1, short __cfwr_p2) {
        for (int __cfwr_i89 = 0; __cfwr_i89 < 7; __cfwr_i89++) {
            for (int __cfwr_i87 = 0; __cfwr_i87 < 6; __cfwr_i87++) {
            try {
            for (int __cfwr_i83 = 0; __cfwr_i83 < 9; __cfwr_i83++) {
            try {
            for (int __cfwr_i63 = 0; __cfwr_i63 < 6; __cfwr_i63++) {
            for (int __cfwr_i48 = 0; __cfwr_i48 < 5; __cfwr_i48++) {
            short __cfwr_item49 = null;
        }
        }
        } catch (Exception __cfwr_e38) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e77) {
            // ignore
        }
        }
        }
        for (int __cfwr_i30 = 0; __cfwr_i30 < 6; __cfwr_i30++) {
            if (true && false) {
            float __cfwr_temp23 = -2.67f;
        }
        }
        return -569L;
    }
    public static boolean __cfwr_calc868(int __cfwr_p0, Boolean __cfwr_p1) {
        for (int __cfwr_i94 = 0; __cfwr_i94 < 6; __cfwr_i94++) {
            try {
            short __cfwr_entry68 = null;
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        }
        return null;
        long __cfwr_val1 = -583L;
        return null;
        return ('x' / -11.47f);
    }
    protected static Object __cfwr_temp733() {
        if (true && true) {
            try {
            for (int __cfwr_i49 = 0; __cfwr_i49 < 1; __cfwr_i49++) {
            try {
            return null;
        } catch (Exception __cfwr_e96) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e91) {
            // ignore
        }
        }
        while (false) {
            for (int __cfwr_i97 = 0; __cfwr_i97 < 8; __cfwr_i97++) {
            while (true) {
            for (int __cfwr_i71 = 0; __cfwr_i71 < 7; __cfwr_i71++) {
            while (false) {
            float __cfwr_data82 = -78.37f;
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}