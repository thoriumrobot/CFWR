import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class ParserOffsetTest {

    public void multiplication1(String[] a, int i, @Positive int j) {
        if ((i * j) < (a.length + j)) {
            @IndexFor("a")
            int k = i;
            @IndexFor("a")
            int k1 = j;
        }
    }
}
