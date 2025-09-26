/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test, @LTLengthOf("arr") int a, @LTLengthOf("arr") int a3) {
        byte __cfwr_obj34 = null;

        Float __cfwr_node42 = null;

    int b = 2;
    if (b < test) {
      @LTLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTLengthOf("arr") int c1 = b;

    if (b < a3) {
      int potato = 7;
    } else {
      // :: error: (assignment)
      @LTLengthOf("arr") int d = b;
    }
  }

  void testLTEL(@LTLengthOf("arr") int test) {
    // :: error: (assignment)
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @LTEqLengthOf("arr") int a3 = Integer.parseInt("3");

    int b = 2;
    if (b < test) {
      @LTEqLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTEqLengthOf("arr") int c1 = b;

    if (b < a) {
      int potato = 7;
    } else {
      // :: error: (assignment)
      @LTEqLengthOf("arr") int d = b;
    }
      String __cfwr_handle502(Object __cfwr_p0, boolean __cfwr_p1, Object __cfwr_p2) {
        for (int __cfwr_i80 = 0; __cfwr_i80 < 7; __cfwr_i80++) {
            for (int __cfwr_i44 = 0; __cfwr_i44 < 7; __cfwr_i44++) {
            int __cfwr_data41 = 113;
        }
        }
        Long __cfwr_temp54 = null;
        Character __cfwr_entry2 = null;
        try {
            try {
            return null;
        } catch (Exception __cfwr_e46) {
            // ignore
        }
        } catch (Exception __cfwr_e57) {
            // ignore
        }
        return "test74";
    }
}
