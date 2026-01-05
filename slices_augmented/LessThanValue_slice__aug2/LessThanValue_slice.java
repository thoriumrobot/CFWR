/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanValue_slice {
    @Positive
  void subtyping(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
        long __cfwr_entry76 = 112L;

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

    private Integer __cfwr_util44(Double __cfwr_p0, boolean __cfwr_p1) {
        for (int __cfwr_i85 = 0; __cfwr_i85 < 5; __cfwr_i85++) {
            try {
            Character __cfwr_node6 = null;
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        }
        try {
            for (int __cfwr_i71 = 0; __cfwr_i71 < 1; __cfwr_i71++) {
            String __cfwr_val65 = "world18";
        }
        } catch (Exception __cfwr_e86) {
            // ignore
        }
        for (int __cfwr_i2 = 0; __cfwr_i2 < 4; __cfwr_i2++) {
            while (false) {
            Long __cfwr_node6 = null;
            break; // Prevent infinite loops
        }
        }
        return null;
    }
    protected Integer __cfwr_func652() {
        try {
            byte __cfwr_result63 = null;
        } catch (Exception __cfwr_e73) {
            // ignore
        }
        for (int __cfwr_i61 = 0; __cfwr_i61 < 8; __cfwr_i61++) {
            while ((81.59f | ('D' % -508L))) {
            Object __cfwr_entry31 = null;
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i25 = 0; __cfwr_i25 < 2; __cfwr_i25++) {
            String __cfwr_data18 = "data15";
        }
        return null;
    }
    public static Double __cfwr_proc877(Integer __cfwr_p0) {
        Integer __cfwr_obj84 = null;
        while ((null ^ 75.91f)) {
            try {
            if (true && true) {
            try {
            return null;
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e72) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        if (false || (232L - (null << 'D'))) {
            while (false) {
            for (int __cfwr_i50 = 0; __cfwr_i50 < 3; __cfwr_i50++) {
            return 324;
        }
            break; // Prevent infinite loops
        }
        }
        return null;
    }
}