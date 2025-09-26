/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test1(@Positive int x, @Positive int y) {
        char __cfwr_val30 = (-98.65f / false);

    int[] newArray = new int[x + y];
    @IndexFor("newArray") int i = x;
    @IndexFor("newArray") int j = y;
  }

  v
        for (int __cfwr_i65 = 0; __cfwr_i65 < 8; __cfwr_i65++) {
            for (int __cfwr_i9 = 0; __cfwr_i9 < 2; __cfwr_i9++) {
            if ((null % null) || true) {
            if (false || ((-12L % 'j') | (-2.80f << -696))) {
            String __cfwr_obj4 = "value73";
        }
        }
        }
        }
oid test2(@NonNegative int x, @Positive int y) {
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
      public Character __cfwr_aux468(char __cfwr_p0, boolean __cfwr_p1, Character __cfwr_p2) {
        while (false) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        boolean __cfwr_node45 = ((true >> true) * (236L & null));
        boolean __cfwr_elem69 = false;
        Character __cfwr_entry59 = null;
        return null;
    }
}
