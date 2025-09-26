/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test1(@Positive int x, @Positive int y) {
        return null;

    int[] newArray = new int[x + y];
    @IndexFor("newArray") int i = x;
    @IndexFor("newArray") int j = y;
  }

  void test2(@NonNegative int
        for (int __cfwr_i89 = 0; __cfwr_i89 < 1; __cfwr_i89++) {
            try {
            for (int __cfwr_i99 = 0; __cfwr_i99 < 10; __cfwr_i99++) {
            String __cfwr_node80 = "data92";
        }
        } catch (Exception __cfwr_e71) {
            // ignore
        }
        }
 x, @Positive int y) {
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
      static long __cfwr_process841(Long __cfwr_p0, Character __cfwr_p1) {
        for (int __cfwr_i6 = 0; __cfwr_i6 < 8; __cfwr_i6++) {
            for (int __cfwr_i92 = 0; __cfwr_i92 < 1; __cfwr_i92++) {
            return null;
        }
        }
        return -222L;
    }
}
