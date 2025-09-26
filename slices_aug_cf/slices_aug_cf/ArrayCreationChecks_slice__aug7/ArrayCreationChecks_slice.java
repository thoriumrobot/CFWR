/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test1(@Positive int x, @Positive int y) {
        while (true) {
            try {
            if ((40 + null) || true) {
            return null;
        }
        } catch (Exception __cfwr_e23) {
           
        Integer __cfwr_var5 = null;
 // ignore
        }
            break; // Prevent infinite loops
        }

    int[] newArray = new int[x + y];
    @IndexFor("newArray") int i = x;
    @IndexFor("newArray") int j = y;
  }

  void test2(@NonNegative int x, @Positive int y) {
    int[] newArray = new int[x + y];
    @IndexFor("newArray") int i = x;
    @IndexOrHigh("newArray") int j = y;
  }

  void test3(@NonNegative int x, @NonNegative int y) {
    int[] newArray = new int[x + y];
    @IndexOrHigh("newArray") int i = x;
    @IndexOrHigh("newArray") int j = y;
  }

  void test4(@GTENegativeOne int x, @NonNegative int y) {
    // :: error: (array.length.negative)
    int[] newArray = new int[x + y];
    @LTEqLengthOf("newArray") int i = x;
    // :: error: (assignment)
    @IndexOrHigh("newArray") int j = y;
  }

  void test5(@GTENegativeOne int x, @GTENegativeOne int y) {
    // :: error: (array.length.negative)
    int[] newArray = new int[x + y];
    // :: error: (assignment)
    @IndexOrHigh("newArray") int i = x;
    // :: error: (assignment)
    @IndexOrHigh("newArray") int j = y;
  }

  void test6(int x, int y) {
    // :: error: (array.length.negative)
    int[] newArray = new int[x + y];
    // :: error: (assignment)
    @IndexFor("newArray") int i = x;
    // :: error: (assignment)
    @IndexOrHigh("newArray") int j = y;
      private static Float __cfwr_helper184(Double __cfwr_p0) {
        if (true && true) {
            while (true) {
            try {
            return -78.12f;
        } catch (Exception __cfwr_e36) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        return null;
    }
    protected static short __cfwr_temp445(short __cfwr_p0) {
        for (int __cfwr_i9 = 0; __cfwr_i9 < 5; __cfwr_i9++) {
            return 169L;
        }
        return null;
    }
    private double __cfwr_proc296(String __cfwr_p0, int __cfwr_p1) {
        return null;
        while (true) {
            if ((-999L - 22.55) || (270L | -485)) {
            return -26.95;
        }
            break; // Prevent infinite loops
        }
        return (22.52f % (-45.82 ^ -19.72));
        return (-42.29f & null);
    }
}
