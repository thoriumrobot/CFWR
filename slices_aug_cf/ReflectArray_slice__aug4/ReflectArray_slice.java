/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testNewInstance(int i) {
        return null;

    // :: error: (argument)
    Array.newInstance(Object.class, i);
    if (i >= 0) {
      Array.newInstance(Object.class, i);
    }
  }

  void testFor(Object a) {
    for (int i = 0; i < Array.getLength(a); ++i) {
      Array.setInt(a, i, 1 + Array.getInt(a, i));
    }
  }

  void testMinLen(Object @MinLen(1) [] a) {
    Array.get(a, 0);
    // :: error: (argument)
    Array.get(a, 1);
      private static char __cfwr_util974(Character __cfwr_p0, String __cfwr_p1) {
        Character __cfwr_val46 = null;
        if (true && true) {
            Long __cfwr_item63 = null;
        }
        Character __cfwr_entry26 = null;
        return 'O';
    }
}
