/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test1(@Positive int x, @Positive int y) {
        try {
            Character __cfwr_data15 = null;
        } catch (Exception __cfwr_e27) {
            // ignore
        }

    int[] newArray = new int[x + y]
        String __cfwr_val63 = "data66";
;
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
      protected static Float __cfwr_proc577(short __cfwr_p0) {
        while (false) {
            return null;
            break; // Prevent infinite loops
        }
        return null;
    }
    public Float __cfwr_proc790(double __cfwr_p0, boolean __cfwr_p1, float __cfwr_p2) {
        if (false || (null - 246L)) {
            byte __cfwr_obj43 = null;
        }
        while (false) {
            while ((62.72 % (-13.65f % 639L))) {
            while ((null * -601L)) {
            while (false) {
            if ((86.94f ^ 66.22f) || ('J' ^ true)) {
            String __cfwr_node38 = "test15";
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    private static char __cfwr_aux129(Character __cfwr_p0, Double __cfwr_p1, boolean __cfwr_p2) {
        Float __cfwr_data40 = null;
        Float __cfwr_result24 = null;
        return '0';
    }
}
