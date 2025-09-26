/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test1(@Positive int x, @Positive int y) {
        if ((null * 299L) && true) {
            while (false) {
            try {
            if (false || true) {
            try {
            Long __cfwr_var77 = null;
        } catch (Exception __cfwr_e24) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e36) {
            // ignore
        }
            break; // Prevent infinite loops
        }
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
      public static Float __cfwr_temp168(Object __cfwr_p0, Double __cfwr_p1) {
        long __cfwr_entry68 = 508L;
        return null;
        return null;
        String __cfwr_data77 = "world4";
        return null;
    }
}
