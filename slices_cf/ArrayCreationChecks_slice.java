  void test1(@Positive int x, @Positive int y) {
    int[] newArray = new int[x + y];
    @NonNegative
    @IndexFor("newArray") int i = x;
    @NonNegative
    @IndexFor("newArray") int j = y;
  }

  void test2(@NonNegative int x, @Positive int y) {
    int[] newArray = new int[x + y];
    @NonNegative
    @IndexFor("newArray") int i = x;
    @NonNegative
    @IndexOrHigh("newArray") int j = y;
  }

  void test3(@NonNegative int x, @NonNegative int y) {
    int[] newArray = new int[x + y];
    @NonNegative
    @IndexOrHigh("newArray") int i = x;
    @NonNegative
    @IndexOrHigh("newArray") int j = y;
  }
